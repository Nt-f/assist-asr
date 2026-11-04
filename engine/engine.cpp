// asr-engine: Qwen3-ASR through llama.cpp's library, for assist-asr's light version.
//
// llama-server encodes the audio again in every slot and returns no scores for prompt text, so its name
// check generated each candidate under a grammar, one after another (~0.7 s on a CPU). Here the audio is
// encoded once into sequence 0, every request copies that (free in llama.cpp's unified cache), and all
// candidates of a name check are scored in one batched forward pass.
//
//   asr-engine -m model.gguf --mmproj mmproj.gguf [--port 8080] [--host 127.0.0.1] [-t threads] [-ngl 99] [-c 8192]
//
// POST /generate {wav, prompt, text?, n_probs?, max_tokens?}
//   prompt: the full prompt up to and including "<asr_text>", with the media marker (GET /props) where
//   the audio goes; text: forced continuation. Greedy decoding from there; answers
//   {"completion_probabilities": [{"token", "logprob", "top_logprobs": [{"token", "logprob"}]}]} as
//   llama-server does (the end of the transcript as the token "").
// POST /score {wav, prompt, given?, tails: [...], end?: true}
//   The summed raw logprob of each tail after prompt + given (with the end of the transcript when end):
//   {"scores": [...]}. One batched pass for all tails.
// GET /props, /health.
//
// The last audio + prompt stays encoded in sequence 0: /generate and /score on the same request reuse it.

#include "ggml-backend.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "httplib.h"
#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;

static llama_model * g_model;
static llama_context * g_ctx;
static mtmd_context * g_mtmd;
static const llama_vocab * g_vocab;
static int g_n_vocab;
static int g_max_seq;
static std::mutex g_lock;

// sequence 0: the session (audio + prompt), as of g_key
static std::string g_key;
static llama_pos g_n_past0 = 0;
// the text before the audio (system turn: HA's names, the vocabulary), the same request after request:
// kept in sequence 0's first positions, as llama-server's prompt cache keeps it (re-reading its ~450
// tokens took a few hundred ms a request on a CPU)
static std::vector<llama_token> g_pre;
static llama_pos g_pre_pos = 0;
static std::vector<float> g_last0;  // logits after the prompt's last token

static std::string b64decode(const std::string & in) {
    static int t[256];
    static bool init = false;
    if (!init) {
        std::fill(t, t + 256, -1);
        const char * a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int i = 0; i < 64; i++) t[(unsigned char) a[i]] = i;
        init = true;
    }
    std::string out;
    out.reserve(in.size() * 3 / 4);
    int val = 0, bits = -8;
    for (unsigned char c : in) {
        if (t[c] < 0) continue;
        val = (val << 6) + t[c];
        bits += 6;
        if (bits >= 0) {
            out.push_back(char((val >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return out;
}

static std::vector<llama_token> tokenize(const std::string & text) {
    if (text.empty()) return {};
    int n = -llama_tokenize(g_vocab, text.c_str(), text.size(), nullptr, 0, false, true);
    std::vector<llama_token> out(n);
    llama_tokenize(g_vocab, text.c_str(), text.size(), out.data(), n, false, true);
    return out;
}

static std::string piece(llama_token t) {
    char buf[256];
    int n = llama_token_to_piece(g_vocab, t, buf, sizeof(buf), 0, true);
    return n > 0 ? std::string(buf, n) : std::string();
}

// log of the softmax's denominator
static float log_norm(const float * logits) {
    float mx = *std::max_element(logits, logits + g_n_vocab);
    double s = 0;
    for (int i = 0; i < g_n_vocab; i++) s += std::exp(double(logits[i] - mx));
    return mx + float(std::log(s));
}

// The audio + prompt into sequence 0, unless already there.
static void session(const std::string & wav, const std::string & prompt) {
    std::string key = std::to_string(std::hash<std::string>{}(wav)) + ":" + std::to_string(std::hash<std::string>{}(prompt));
    if (key == g_key) return;
    llama_memory_t mem = llama_get_memory(g_ctx);
    g_key.clear();

    mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_buf(
        g_mtmd, (const unsigned char *) wav.data(), wav.size(), false, mtmd_helper_init_opt_default());
    if (!bw.bitmap) throw std::runtime_error("audio not understood (16 kHz mono WAV expected)");
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    mtmd_input_text txt = {prompt.c_str(), prompt.size(), true, true};
    const mtmd_bitmap * bms[1] = {bw.bitmap};
    int32_t r = mtmd_tokenize(g_mtmd, chunks, &txt, bms, 1);
    if (r != 0) {
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bw.bitmap);
        throw std::runtime_error("prompt and audio do not tokenize (media marker missing?)");
    }
    // keep what stays: the leading text chunk when it is the same as last time's
    size_t n_chunks = mtmd_input_chunks_size(chunks), first = 0;
    llama_pos n_past = 0;
    const mtmd_input_chunk * c0 = n_chunks ? mtmd_input_chunks_get(chunks, 0) : nullptr;
    std::vector<llama_token> pre;
    if (c0 && mtmd_input_chunk_get_type(c0) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        size_t n = 0;
        const llama_token * t = mtmd_input_chunk_get_tokens_text(c0, &n);
        pre.assign(t, t + n);
    }
    for (int s = 1; s < g_max_seq; s++) llama_memory_seq_rm(mem, s, -1, -1);
    if (!pre.empty() && pre == g_pre && n_chunks > 1) {
        llama_memory_seq_rm(mem, 0, g_pre_pos, -1);
        n_past = g_pre_pos;
        first = 1;
    } else {
        llama_memory_seq_rm(mem, 0, -1, -1);
        g_pre.clear();
    }
    r = 0;
    auto t0 = std::chrono::steady_clock::now();
    llama_pos kept = n_past;
    for (size_t i = first; i < n_chunks && r == 0; i++) {
        r = mtmd_helper_eval_chunk_single(g_mtmd, g_ctx, mtmd_input_chunks_get(chunks, i), n_past, 0,
                                          llama_n_batch(g_ctx), i + 1 == n_chunks, &n_past);
        if (i == 0 && r == 0 && !pre.empty()) {
            g_pre = pre;
            g_pre_pos = n_past;
        }
    }
    fprintf(stderr, "asr-engine: session: %d positions kept, %d evaluated in %.0f ms\n", (int) kept,
            (int) (n_past - kept), std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bw.bitmap);
    if (r != 0) {
        g_pre.clear();
        throw std::runtime_error("evaluating the prompt failed");
    }
    const float * l = llama_get_logits_ith(g_ctx, -1);
    g_last0.assign(l, l + g_n_vocab);
    g_n_past0 = n_past;
    g_key = key;
}

// Sequence seq := sequence 0 + tokens; returns the logits after the last (those of 0 when no tokens).
static std::vector<float> extend(llama_seq_id seq, const std::vector<llama_token> & toks, llama_pos & n_past) {
    llama_memory_t mem = llama_get_memory(g_ctx);
    llama_memory_seq_rm(mem, seq, -1, -1);
    llama_memory_seq_cp(mem, 0, seq, -1, -1);
    n_past = g_n_past0;
    if (toks.empty()) return g_last0;
    llama_batch b = llama_batch_init(toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); i++) {
        b.token[i] = toks[i];
        b.pos[i] = n_past + i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = seq;
        b.logits[i] = i + 1 == toks.size();
    }
    b.n_tokens = toks.size();
    int r = llama_decode(g_ctx, b);
    llama_batch_free(b);
    if (r != 0) throw std::runtime_error("decode failed");
    n_past += toks.size();
    const float * l = llama_get_logits_ith(g_ctx, -1);
    return std::vector<float>(l, l + g_n_vocab);
}

static json generate(const json & q) {
    session(b64decode(q.at("wav").get<std::string>()), q.at("prompt").get<std::string>());
    int n_probs = q.value("n_probs", 0), max_tokens = q.value("max_tokens", 200);
    llama_pos n_past;
    std::vector<float> logits = extend(1, tokenize(q.value("text", "")), n_past);
    json steps = json::array();
    llama_batch b = llama_batch_init(1, 0, 1);
    for (int i = 0; i < max_tokens; i++) {
        float norm = log_norm(logits.data());
        int best = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
        json step = {{"token", llama_vocab_is_eog(g_vocab, best) ? std::string() : piece(best)},
                     {"logprob", logits[best] - norm}};
        if (n_probs > 0) {
            // the n_probs highest in one pass with a small min-heap (sorting all 151k cost ~2 ms a token)
            std::vector<int> idx;
            idx.reserve(n_probs + 1);
            auto worse = [&](int a, int c) { return logits[a] > logits[c]; };
            for (int k = 0; k < g_n_vocab; k++) {
                if ((int) idx.size() < n_probs) {
                    idx.push_back(k);
                    std::push_heap(idx.begin(), idx.end(), worse);
                } else if (logits[k] > logits[idx.front()]) {
                    std::pop_heap(idx.begin(), idx.end(), worse);
                    idx.back() = k;
                    std::push_heap(idx.begin(), idx.end(), worse);
                }
            }
            std::sort_heap(idx.begin(), idx.end(), worse);
            json top = json::array();
            for (int k = 0; k < (int) idx.size(); k++)
                top.push_back({{"token", llama_vocab_is_eog(g_vocab, idx[k]) ? std::string() : piece(idx[k])},
                               {"logprob", logits[idx[k]] - norm}});
            step["top_logprobs"] = top;
        }
        steps.push_back(step);
        if (llama_vocab_is_eog(g_vocab, best)) break;
        b.token[0] = best;
        b.pos[0] = n_past++;
        b.n_seq_id[0] = 1;
        b.seq_id[0][0] = 1;
        b.logits[0] = 1;
        b.n_tokens = 1;
        if (llama_decode(g_ctx, b) != 0) {
            llama_batch_free(b);
            throw std::runtime_error("decode failed");
        }
        const float * l = llama_get_logits_ith(g_ctx, -1);
        logits.assign(l, l + g_n_vocab);
    }
    llama_batch_free(b);
    return {{"completion_probabilities", steps}};
}

static json score(const json & q) {
    session(b64decode(q.at("wav").get<std::string>()), q.at("prompt").get<std::string>());
    bool end = q.value("end", true);
    std::vector<std::string> tails = q.at("tails").get<std::vector<std::string>>();
    if ((int) tails.size() > g_max_seq - 2) throw std::runtime_error("too many tails");
    // sequence 1 = prompt + given; each tail in a sequence of its own, copied from it
    llama_pos n_given;
    std::vector<float> given_last = extend(1, tokenize(q.value("given", "")), n_given);
    float given_norm = log_norm(given_last.data());
    llama_memory_t mem = llama_get_memory(g_ctx);
    llama_token eot = llama_vocab_eot(g_vocab) != LLAMA_TOKEN_NULL ? llama_vocab_eot(g_vocab) : llama_vocab_eos(g_vocab);
    {   // "<|im_end|>" ends a Qwen transcript: the token to score the end with
        std::vector<llama_token> im = tokenize("<|im_end|>");
        if (im.size() == 1) eot = im[0];
    }
    std::vector<std::vector<llama_token>> toks;
    int total = 0;
    for (size_t i = 0; i < tails.size(); i++) {
        toks.push_back(tokenize(tails[i]));
        total += toks.back().size();
        llama_memory_seq_rm(mem, 2 + i, -1, -1);
        llama_memory_seq_cp(mem, 1, 2 + i, -1, -1);
    }
    std::vector<double> scores(tails.size(), 0.0);
    if (total > 0) {
        // all tails in one batch; every token's logits are kept: they score the token after it
        llama_batch b = llama_batch_init(total, 0, 1);
        int k = 0;
        for (size_t i = 0; i < tails.size(); i++)
            for (size_t j = 0; j < toks[i].size(); j++, k++) {
                b.token[k] = toks[i][j];
                b.pos[k] = n_given + j;
                b.n_seq_id[k] = 1;
                b.seq_id[k][0] = 2 + i;
                b.logits[k] = 1;
            }
        b.n_tokens = total;
        int r = llama_decode(g_ctx, b);
        llama_batch_free(b);
        if (r != 0) throw std::runtime_error("decode failed");
        k = 0;
        for (size_t i = 0; i < tails.size(); i++) {
            const auto & t = toks[i];
            if (t.empty()) continue;
            double s = given_last[t[0]] - given_norm;  // the first token, after the given text
            for (size_t j = 0; j < t.size(); j++, k++) {
                const float * l = llama_get_logits_ith(g_ctx, k);
                bool last = j + 1 == t.size();
                if (!last || end) {
                    llama_token next = last ? eot : t[j + 1];
                    s += l[next] - log_norm(l);
                }
            }
            scores[i] = s;
        }
    }
    for (size_t i = 0; i < tails.size(); i++) llama_memory_seq_rm(mem, 2 + i, -1, -1);
    return {{"scores", scores}};
}

int main(int argc, char ** argv) {
    std::string model, mmproj, host = "127.0.0.1";
    int port = 8080, threads = std::max(1u, std::thread::hardware_concurrency() / 2), ngl = 0, n_ctx = 8192;
    g_max_seq = 16;
    bool unified = true;
    int n_ubatch = 512;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "-m") model = next();
        else if (a == "--mmproj") mmproj = next();
        else if (a == "--host") host = next();
        else if (a == "--port") port = std::stoi(next());
        else if (a == "-t") threads = std::stoi(next());
        else if (a == "-ngl") ngl = std::stoi(next());
        else if (a == "-c") n_ctx = std::stoi(next());
        else if (a == "--seqs") g_max_seq = std::stoi(next());
        else if (a == "--no-unified") unified = false;
        else if (a == "-ub") n_ubatch = std::stoi(next());
    }
    if (model.empty() || mmproj.empty()) {
        fprintf(stderr, "usage: asr-engine -m model.gguf --mmproj mmproj.gguf [--port N] [--host H] [-t N] [-ngl N] [-c N]\n");
        return 1;
    }
    ggml_backend_load_all();
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    g_model = llama_model_load_from_file(model.c_str(), mp);
    if (!g_model) return 1;
    g_vocab = llama_model_get_vocab(g_model);
    g_n_vocab = llama_vocab_n_tokens(g_vocab);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx;
    cp.n_batch = 2048;
    cp.n_ubatch = n_ubatch;
    cp.n_seq_max = g_max_seq;
    cp.kv_unified = unified;  // the candidates share sequence 0's cells instead of copies
    cp.n_threads = cp.n_threads_batch = threads;
    g_ctx = llama_init_from_model(g_model, cp);
    if (!g_ctx) return 1;
    // a thread pool kept for the engine's life, as llama-server has (else ggml makes threads per call)
    if (ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
        auto * reg = ggml_backend_dev_backend_reg(cpu);
        auto * tp_new = (ggml_threadpool_t (*)(ggml_threadpool_params *)) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
        if (tp_new) {
            ggml_threadpool_params tpp = ggml_threadpool_params_default(threads);
            if (ggml_threadpool_t tp = tp_new(&tpp)) {
                llama_attach_threadpool(g_ctx, tp, tp);
                fprintf(stderr, "asr-engine: thread pool of %d attached\n", threads);
            }
        }
    }
    mtmd_context_params tp = mtmd_context_params_default();
    tp.use_gpu = ngl > 0;
    tp.n_threads = threads;
    tp.print_timings = false;
    g_mtmd = mtmd_init_from_file(mmproj.c_str(), g_model, tp);
    if (!g_mtmd) return 1;

    httplib::Server srv;
    auto handle = [](json (*fn)(const json &)) {
        return [fn](const httplib::Request & req, httplib::Response & res) {
            std::lock_guard<std::mutex> lk(g_lock);
            try {
                // a token is a byte piece and may cut a character in two: those bytes are replaced
                res.set_content(fn(json::parse(req.body)).dump(-1, ' ', false, json::error_handler_t::replace),
                                "application/json");
            } catch (const std::exception & e) {
                g_key.clear();  // whatever state sequence 0 is in now, it is not the session's
                res.status = 400;
                res.set_content(json{{"error", e.what()}}.dump(), "application/json");
            }
        };
    };
    srv.Post("/generate", handle(generate));
    srv.Post("/score", handle(score));
    srv.Get("/props", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(json{{"media_marker", mtmd_default_marker()}}.dump(), "application/json");
    });
    srv.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    fprintf(stderr, "asr-engine: listening on %s:%d (%d threads, %d layers on the GPU)\n", host.c_str(), port, threads, ngl);
    srv.listen(host, port);
    return 0;
}
