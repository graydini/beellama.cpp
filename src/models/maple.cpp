#include "models.h"

// Maple-Preview (DeepGrove, MIT): 20B-A1B ternary-weight MoE reasoning LLM.
//
// Block structure (matches the MLX reference implementation):
//   - 3:1 hybrid attention: sliding-window-512 layers (partial RoPE, 64/128 dims)
//     interleaved with full-attention layers that carry NO RoPE (n_rot_full == 0),
//     full attention at il % 4 == 3.
//   - Flash-head QK: per-head RMSNorm (attn_q_norm / attn_k_norm) applied on the
//     reshaped Q/K tensors before RoPE, k_proj at n_embd_gqa (4 heads x 128).
//   - MoE: 256 experts, 8 active, moe_intermediate 512, clamp-7 SwiGLU
//     (silu(min(gate, +7)) * clip(up, -7, +7)) with fp32 softmax + renorm routing.

void llama_model_maple::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // 3:1 SWA:GA hybrid; full-attention layers at il % 4 == 3 (config layer_types).
    hparams.set_swa_pattern(4, false);
    // Sliding window for the SWA layers (maple.attention.sliding_window in the GGUF).
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
    // Sliding-window attention cache (windowed KV, window = 512 from the GGUF).
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;

    // GGUF has no rope.type key; Maple uses plain interleaved RoPE, supplied by
    // llama_model_rope_type() via the LLM_ARCH_MAPLE -> LLAMA_ROPE_TYPE_NEOX case.

    // Clamp-7 SwiGLU for the routed experts (consumed by build_moe_ffn).
    for (uint32_t i = 0; i < hparams.n_layer(); ++i) {
        hparams.swiglu_clamp_exp[i] = 7.0f;
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_maple::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    // if output is NULL, init from the input tok embed
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        // Flash-head QK: per-head RMSNorm
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);

        if (n_expert == 0) {
            throw std::runtime_error("n_expert must be > 0 for MAPLE");
        }
        if (n_expert_used == 0) {
            throw std::runtime_error("n_expert_used must be > 0 for MAPLE");
        }

        // MoE branch
        const int64_t n_ff_exp = hparams.n_ff_exp(i) ? hparams.n_ff_exp(i) : n_ff / n_expert_used;

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_maple::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_maple::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self_attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            // Flash-head QK: per-head RMSNorm, then partial RoPE.
            // n_rot(il) is 64 on SWA layers and 0 on full-attention layers,
            // so the full-attention layers get a no-op RoPE (no position info).
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    hparams.n_rot(il), rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    hparams.n_rot(il), rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // MoE branch (clamp-7 SwiGLU handled inside build_moe_ffn for MAPLE)
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * moe_out =
            build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, true,
                    hparams.expert_weights_scale,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il,
                    nullptr, nullptr,
                    model.layers[il].ffn_up_exps_s,
                    model.layers[il].ffn_gate_exps_s,
                    model.layers[il].ffn_down_exps_s);
        cb(moe_out, "ffn_moe_out", il);
        cur = moe_out;

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
