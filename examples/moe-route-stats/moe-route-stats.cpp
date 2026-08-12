// moe-route-stats - measure which experts a MoE model actually routes to.
//
// Why this exists: deciding whether saliency-guided expert placement can help a
// spill-bound MoE requires knowing how CONCENTRATED route mass is. If a small
// set of experts carries most of the routing, pinning that set in fast memory
// collapses the slow tail. If routing is flat, capacity sets the ceiling and no
// placement policy helps. That is a go/no-go question, and answering it needs
// per-expert statistics from the model as actually served - i.e. from the GGUF,
// not from an HF checkpoint one may not have.
//
// No core patch is needed: llama.cpp already names the expert-selection tensor
// "ffn_moe_topk" via cb(), and already exposes an eval callback. This observes
// that tensor, accumulates a per-layer histogram of selected expert ids, and
// reports the concentration curve.
//
// Usage:
//   llama-moe-route-stats -m model.gguf -f prompts.txt [-ngl N] [--json out.json]
//
// prompts.txt is one prompt per line; each is evaluated (prompt processing only,
// which is where the bulk of routing decisions happen) and the routes counted.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdio>
#include <fstream>
#include <map>
#include <numeric>
#include <string>
#include <cstring>
#include <vector>

struct route_stats {
    // layer -> (expert id -> times selected)
    std::map<int, std::vector<int64_t>> per_layer;
    // layer -> streaming first and second moments of the ROUTER LOGIT vector.
    // With mean mu and covariance S, "can expert i out-score expert j anywhere
    // within r standard deviations of the observed logit distribution" has a
    // closed form:
    //     (mu_i - mu_j) + r*sqrt(S_ii + S_jj - 2*S_ij) > 0
    // because the support function of an ellipsoid along direction (e_i - e_j)
    // is exactly that. An expert that cannot out-score at least (n_expert - k)
    // others can never enter the top-k, so it is unreachable on that region.
    // This is the reachability question a compiler asks about a branch, and it
    // is strictly better than counting selections: an expert that simply was
    // not sampled still reads as REACHABLE if its logits overlap the top-k band.
    std::map<int, std::vector<double>> logit_sum;      // [n_expert]
    std::map<int, std::vector<double>> logit_sumsq;    // [n_expert*n_expert], row-major
    std::map<int, int64_t> logit_tokens;
    // Running max of (l_i - l_j) over observed tokens, [n_expert*n_expert].
    // This is the EXACT support function of the convex hull of the observed
    // logits: sup over conv(S) of (e_i - e_j).l == max over S of (l_i - l_j),
    // because a linear functional attains its max on a hull at a vertex. So the
    // hull test costs one pass and certifies a whole continuous convex region,
    // where an ellipsoid only models one. Prefer this: LLM hidden states are
    // badly anisotropic (massive activations, outlier dims), so a
    // covariance-fitted ellipsoid inflates along a few rogue directions and
    // certifies almost nothing.
    std::map<int, std::vector<float>> logit_maxdiff;
    // INTRA-EXPERT reachability. The same argument one level down: inside an
    // expert FFN the neuron contributes silu(gate_n(x)) * up_n(x), and gate_n is
    // a linear functional of the input, so "can neuron n ever activate on this
    // workload" is again a support function. On the convex hull that is just a
    // running max. Reported literature claims up to 90% neuron-level sparsity
    // inside experts, which is where there may actually be something to remove -
    // the expert level came back empty.
    std::map<int, std::vector<float>> gate_max;   // [n_ff] running max per layer
    std::map<int, char> gate_biased;              // prefer the post-bias tensor
    std::map<int, char> logit_biased;             // same, for router scores
    std::map<std::string, int> seen_names;        // diagnostic: what actually fires
    bool collect_gate = false;                    // expensive; opt in separately
    std::vector<uint8_t> scratch;
    std::vector<float>   lscratch;
    int64_t total_selections = 0;
    int n_expert_guess = 0;

    // SELF-CHECK. Everything downstream assumes the score tensor we accumulate is
    // the one the router actually compared. That assumption is silent, and when it
    // broke on DeepSeek it produced confident wrong numbers rather than an error.
    // So: stash the scores last seen for a layer, and when ffn_moe_topk arrives for
    // that same ubatch, recompute the top-k OURSELVES and compare it to what the
    // model actually chose. Any disagreement means we are watching the wrong
    // tensor - e.g. a model whose selection bias is applied in probability space
    // (probs + exp_probs_b) rather than logit space, which no amount of staring at
    // logit tensors will reveal.
    std::map<int, std::vector<float>> last_scores;   // layer -> [ne*nt] copy
    std::map<int, int>     last_ne;
    std::map<int, int>     last_nt;
    // The score node does not reliably execute before ffn_moe_topk (cb() renames
    // tensors, so which node carries the selection score is arch-dependent and so
    // is its position in the graph). Buffer the selection too and compare from
    // whichever side arrives second.
    std::map<int, std::vector<int32_t>> last_topk;   // layer -> [kk*nt] compacted
    std::map<int, int>     last_kk;
    std::map<int, int>     last_topk_nt;
    // ELECTION. Rather than hardcoding which tensor name carries the selection
    // score - which differs per architecture and which we got wrong twice - try
    // every candidate on the first ubatch of each layer and keep the one that
    // actually reproduces the model's top-k. The instrument calibrates itself.
    std::map<int, std::string> chosen_score;                        // layer -> winning name
    std::map<int, std::map<std::string, std::vector<float>>> cand;  // layer -> name -> scores
    std::map<int, int> cand_ne, cand_nt;
    std::map<int, double> elect_agree;                              // winning agreement rate
    // WAVEGAUGE: optional per-token top-k sequence dump (MOE_TOPK_DUMP=<path>).
    // The aggregate histograms cannot answer temporal questions - how sticky the
    // expert set is token-to-token is exactly what decides whether any prefetch
    // or residency cache can work, and it is the baseline an MTP-guided
    // predictor must beat. layer -> flat [k * T] ids, in decode order.
    std::map<int, std::vector<int32_t>> topk_seq;
    std::map<int, int> topk_k;
    std::vector<int64_t> prompt_bounds;   // per-layer token index at each prompt end
    bool dump_topk = false;
    std::map<int, int64_t> check_tokens;             // tokens compared
    std::map<int, int64_t> check_mismatch;           // non-tie-equivalent set differences
    std::map<int, int64_t> check_tie_equivalent;     // different ids, valid tied top-k
    // Some architectures route early layers by token-id lookup rather than a
    // learned score (DeepSeek-V4's ffn_gate_tid2eid table). Those selections are
    // valid for traffic histograms but have no continuous router score and must
    // never enter score-based reachability analysis.
    std::map<int, char>    hash_layers;
    std::map<int, char>    noncontig;                // score tensor was not contiguous
    int mismatch_reports = 0;
};

// Pull the layer index out of a tensor name like "ffn_moe_topk-23".
static int layer_of(const char * name) {
    const char * dash = strrchr(name, '-');
    if (!dash) return -1;
    return atoi(dash + 1);
}

// Decide WHICH observed tensor is the score the router compares, by testing each
// candidate against the selection the model actually made. Hardcoding the name is
// what failed: cb() renames tensors, so the node carrying the selection score is
// called "ffn_moe_probs" on gpt-oss, "ffn_moe_probs_biased" on DeepSeek, and
// "ffn_moe_logits" only when nothing is applied after the matmul.
static void try_elect(route_stats * st, int il) {
    auto ci = st->cand.find(il);
    auto ti = st->last_topk.find(il);
    if (ci == st->cand.end() || ti == st->last_topk.end()) return;
    const int ne = st->cand_ne[il], nt = st->cand_nt[il];
    const int kk = st->last_kk[il], ntk = st->last_topk_nt[il];
    if (nt != ntk || ne <= 0 || kk <= 0 || kk > ne) return;

    const int32_t * T = ti->second.data();
    // On a tie, prefer the tensor CLOSEST to the top_k. A router bias often does
    // not change the top-k on a short election ubatch, so the raw pre-bias logits
    // can reproduce it perfectly and then diverge later. Ties are common and must
    // break toward the most-derived score, not alphabetically.
    auto pref = [](const std::string & n) {
        if (n == "ffn_moe_probs_masked")  return 5;
        if (n == "ffn_moe_probs_biased")  return 4;
        if (n == "ffn_moe_probs")         return 3;
        if (n == "ffn_moe_logits_biased") return 2;
        return 1;                          // ffn_moe_logits
    };
    std::string best_name; double best_rate = -1.0; int best_pref = -1;
    for (auto & [nm, V] : ci->second) {
        int ok = 0;
        std::vector<int> idx(ne), mine, theirs;
        for (int tk = 0; tk < nt; ++tk) {
            const float * v = V.data() + (size_t) tk * ne;
            for (int i = 0; i < ne; ++i) idx[i] = i;
            std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                              [&](int a, int b) { return v[a] != v[b] ? v[a] > v[b] : a < b; });
            mine.assign(idx.begin(), idx.begin() + kk);
            theirs.clear();
            for (int j = 0; j < kk; ++j) {
                const int32_t e = T[(size_t) tk * kk + j];
                if (e >= 0) theirs.push_back((int) e);
            }
            std::sort(mine.begin(), mine.end());
            std::sort(theirs.begin(), theirs.end());
            if (mine == theirs) ok++;
        }
        const double rate = nt ? (double) ok / (double) nt : 0.0;
        const int    pr   = pref(nm);
        if (rate > best_rate || (rate == best_rate && pr > best_pref)) {
            best_rate = rate; best_pref = pr; best_name = nm;
        }
    }
    if (best_name.empty()) return;
    st->chosen_score[il] = best_name;
    st->elect_agree[il]  = best_rate;
    if (il <= 1) {
        fprintf(stderr, "[elect] layer %d -> %s (reproduces %.1f%% of selections; "
                "%d candidates)\n", il, best_name.c_str(), 100.0 * best_rate,
                (int) ci->second.size());
    }
    st->cand.erase(il);
    st->last_topk.erase(il);
}

// Compare the top-k we derive from the accumulated score tensor against the one
// the model actually chose, once BOTH have arrived for the same layer/ubatch.
static void try_compare(route_stats * st, int il) {
    auto si = st->last_scores.find(il);
    auto ti = st->last_topk.find(il);
    if (si == st->last_scores.end() || ti == st->last_topk.end()) return;
    const int ne = st->last_ne[il], nt = st->last_nt[il];
    const int kk = st->last_kk[il], ntk = st->last_topk_nt[il];
    if (nt != ntk || ne <= 0 || kk <= 0) { return; }

    const float   * S = si->second.data();
    const int32_t * T = ti->second.data();

    // Accumulate HERE, not on arrival. Several graph nodes can share the elected
    // name (llama.cpp renames tensors), so "the score that produced this
    // selection" is only well defined for a matched pair. Accumulating on arrival
    // double-counted and mixed in a post-top_k tensor.
    {
        auto & mu = st->logit_sum[il];
        auto & sq = st->logit_sumsq[il];
        auto & md = st->logit_maxdiff[il];
        if ((int) mu.size() != ne) { mu.assign(ne, 0.0); sq.assign((size_t) ne * ne, 0.0); }
        if ((int) md.size() != ne * ne) { md.assign((size_t) ne * ne, -std::numeric_limits<float>::infinity()); }
        for (int tk = 0; tk < nt; ++tk) {
            const float * v = S + (size_t) tk * ne;
            for (int i = 0; i < ne; ++i) {
                mu[i] += v[i];
                const double vi = v[i];
                for (int j = i; j < ne; ++j) sq[(size_t) i * ne + j] += vi * v[j];
                float * mdi = md.data() + (size_t) i * ne;
                for (int j = 0; j < ne; ++j) {
                    const float d = v[i] - v[j];
                    if (d > mdi[j]) mdi[j] = d;
                }
            }
        }
        st->logit_tokens[il] += nt;
    }

    std::vector<int> idx(ne), mine, theirs;
    for (int tk = 0; tk < nt; ++tk) {
        const float * v = S + (size_t) tk * ne;
        for (int i = 0; i < ne; ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                          [&](int a, int b) { return v[a] != v[b] ? v[a] > v[b] : a < b; });
        mine.assign(idx.begin(), idx.begin() + kk);
        theirs.clear();
        for (int j = 0; j < kk; ++j) {
            const int32_t e = T[(size_t) tk * kk + j];
            if (e >= 0) theirs.push_back((int) e);
        }
        std::sort(mine.begin(), mine.end());
        std::sort(theirs.begin(), theirs.end());
        st->check_tokens[il]++;
        if (mine != theirs) {
            // Backends need not choose the same expert id at an exact cutoff
            // tie. Validate the mathematical top-k condition: every selected
            // score must be >= every unselected score. This distinguishes a
            // backend tie policy from observing the wrong score tensor.
            std::vector<char> selected(ne, 0);
            for (int e : theirs) if (e >= 0 && e < ne) selected[e] = 1;
            float min_selected = std::numeric_limits<float>::infinity();
            float max_unselected = -std::numeric_limits<float>::infinity();
            for (int e = 0; e < ne; ++e) {
                if (selected[e]) min_selected = std::min(min_selected, v[e]);
                else             max_unselected = std::max(max_unselected, v[e]);
            }
            const bool tie_equivalent =
                (int) theirs.size() == kk && min_selected >= max_unselected;
            if (tie_equivalent) st->check_tie_equivalent[il]++;
            else                st->check_mismatch[il]++;
            if (st->mismatch_reports < 6) {
                st->mismatch_reports++;
                std::vector<int> rk(ne);
                for (int i = 0; i < ne; ++i) rk[i] = i;
                std::sort(rk.begin(), rk.end(), [&](int a, int b) { return v[a] != v[b] ? v[a] > v[b] : a < b; });
                std::vector<int> rank_of(ne, -1);
                for (int r = 0; r < ne; ++r) rank_of[rk[r]] = r;
                fprintf(stderr, "[selfcheck:%s] L%d tok%d predicted {",
                        tie_equivalent ? "cutoff-tie" : "ERROR", il, tk);
                for (size_t j = 0; j < mine.size(); ++j)
                    fprintf(stderr, "%s%d", j ? "," : "", mine[j]);
                fprintf(stderr, "} chose {");
                for (size_t j = 0; j < theirs.size(); ++j)
                    fprintf(stderr, "%s%d", j ? "," : "", theirs[j]);
                fprintf(stderr, "} ranks-in-our-scores {");
                for (size_t j = 0; j < theirs.size(); ++j)
                    fprintf(stderr, "%s%d", j ? "," : "", rank_of[theirs[j]]);
                fprintf(stderr, "} cutoff(selected=%.9g unselected=%.9g)\n",
                        min_selected, max_unselected);
            }
        }
    }
    st->last_scores.erase(il);
    st->last_topk.erase(il);
}

static bool cb_routes(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (route_stats *) user_data;
    if (!ask && strncmp(t->name, "ffn_moe_", 8) == 0) {
        std::string base(t->name);
        const size_t dash = base.rfind('-');
        if (dash != std::string::npos) base = base.substr(0, dash);
        st->seen_names[base]++;
    }

    // Two tensors are wanted: the top-k selection (what actually fired) and the
    // raw router logits (what COULD have fired). Everything else is skipped so
    // the run stays cheap.
    // Models take one of two paths: a MERGED gate_up (one wide tensor, gate in
    // the first half) or SEPARATE gate/up. gpt-oss uses the separate path, so
    // match the gate tensor either way and note whether it is post-bias -- the
    // bias is part of the pre-activation and judging on pre-bias values is wrong.
    const bool is_merged = strncmp(t->name, "ffn_moe_gate_up", 15) == 0
                        && strstr(t->name, "_scaled") == nullptr;
    const bool is_sep    = strncmp(t->name, "ffn_moe_gate", 12) == 0
                        && strstr(t->name, "_up") == nullptr;
    const bool is_gateup = is_merged || is_sep;
    if (is_gateup && !st->collect_gate) return false;
    if (ask && is_gateup) return st->collect_gate;
    if (!ask && is_gateup && t->type == GGML_TYPE_F32) {
        const bool biased = strstr(t->name, "_biased") != nullptr;
        const int il = layer_of(t->name);
        // If a bias is applied later, the pre-bias values are the wrong thing to
        // judge on; restart the max from the biased tensor when it first appears.
        if (biased && !st->gate_biased[il]) { st->gate_biased[il] = 1; st->gate_max[il].clear(); }
        if (!biased && st->gate_biased[il]) return true;
        const size_t nb = ggml_nbytes(t);
        const bool host = ggml_backend_buffer_is_host(t->buffer);
        const float * G;
        if (host) { G = (const float *) t->data; }
        else { st->lscratch.resize(nb / sizeof(float));
               ggml_backend_tensor_get(t, st->lscratch.data(), 0, nb);
               G = st->lscratch.data(); }
        const int wide = (int) t->ne[0];
        const int n_ff = is_merged ? wide / 2 : wide;
        const int64_t rows = ggml_nelements(t) / std::max(1, wide);
        auto & gm = st->gate_max[il];
        if ((int) gm.size() != n_ff) gm.assign(n_ff, -std::numeric_limits<float>::infinity());
        for (int64_t r = 0; r < rows; ++r) {
            const float * v = G + (size_t) r * wide;
            for (int n = 0; n < n_ff; ++n) if (v[n] > gm[n]) gm[n] = v[n];
        }
        return true;
    }

    const bool is_topk   = strncmp(t->name, "ffn_moe_topk",   12) == 0;
    if (is_topk && t->op == GGML_OP_GET_ROWS && t->src[0] != nullptr &&
        strstr(t->src[0]->name, "ffn_gate_tid2eid") != nullptr) {
        const int il = layer_of(t->name);
        st->hash_layers[il] = 1;
        st->cand.erase(il);
        st->last_scores.erase(il);
        st->last_topk.erase(il);
    }
    // DeepSeek-style aux-loss-free load balancing adds a PER-EXPERT BIAS to the
    // score used for selection, so the top-k constraint is affine rather than
    // linear: (w_i - w_j).h > b_j - b_i. Judging on the pre-bias logits would be
    // simply wrong on those models. Accept both and prefer the biased tensor
    // when it appears, mirroring the gate handling below.
    // Which tensor is the score the router ACTUALLY compares? llama.cpp's cb()
    // RENAMES a tensor rather than tagging it, and build_moe_ffn reuses the same
    // tensor object across steps, so the final name is the only reliable marker -
    // and it differs per architecture:
    //   * SOFTMAX_WEIGHT (gpt-oss): probs IS logits(+router bias), renamed to
    //     "ffn_moe_probs". The node still called "ffn_moe_logits" is the RAW
    //     mul_mat, before the bias. Accumulating it silently drops the bias.
    //   * DeepSeek aux-loss-free: selection uses probs + exp_probs_b, named
    //     "ffn_moe_probs_biased" - applied in PROBABILITY space, so it is not
    //     recoverable from logits at all.
    //   * grouped routing: "ffn_moe_probs_masked" after the group mask.
    // Rank them and always accumulate the highest-ranked one seen for a layer.
    // Monotone gating (softmax/sigmoid) preserves pairwise sign, so the hull test
    // is valid in probability space too.
    const int srank =
          strncmp(t->name, "ffn_moe_probs_masked",  20) == 0 ? 5
        : strncmp(t->name, "ffn_moe_probs_biased",  20) == 0 ? 4
        : strncmp(t->name, "ffn_moe_logits_biased", 21) == 0 ? 3
        : strncmp(t->name, "ffn_moe_probs",         13) == 0 ? 2
        : strncmp(t->name, "ffn_moe_logits",        14) == 0 ? 1
        : 0;
    const bool is_logits = srank > 0;
    if (ask) {
        return is_topk || is_logits;
    }
    if (!is_topk && !is_logits) {
        return true;
    }

    if (is_logits && t->type == GGML_TYPE_F32) {
        const int il = layer_of(t->name);
        std::string bn(t->name);
        { const size_t d = bn.rfind('-'); if (d != std::string::npos) bn = bn.substr(0, d); }

        // Read stride-aware: never assume a tensor you did not allocate is packed.
        const int ne = (int) t->ne[0];
        const int nt = (int) (t->ne[1] * t->ne[2] * t->ne[3]);
        // Expert cardinality comes from the full score tensor, not the largest
        // selected id. Otherwise an entirely silent high-numbered tail makes
        // the report understate n_expert and disappear from never-selected
        // counts.
        st->n_expert_guess = std::max(st->n_expert_guess, ne);
        const size_t rs = t->nb[1];
        const size_t span = rs * (size_t) (nt - 1) + (size_t) ne * t->nb[0];
        const bool host = ggml_backend_buffer_is_host(t->buffer);
        const uint8_t * sbase;
        if (host) { sbase = (const uint8_t *) t->data; }
        else { st->scratch.resize(span);
               ggml_backend_tensor_get(t, st->scratch.data(), 0, span);
               sbase = st->scratch.data(); }
        std::vector<float> V((size_t) ne * nt);
        for (int tk = 0; tk < nt; ++tk) {
            memcpy(V.data() + (size_t) tk * ne, sbase + (size_t) tk * rs, (size_t) ne * sizeof(float));
        }

        auto ch = st->chosen_score.find(il);
        if (ch == st->chosen_score.end()) {
            // still electing: remember this candidate and try to decide
            st->cand[il][bn] = V;
            st->cand_ne[il] = ne;
            st->cand_nt[il] = nt;
            try_elect(st, il);
            return true;
        }
        if (bn != ch->second) return true;    // not the elected selection score
        // Several nodes can end up with the same name; the post-top_k weight
        // tensor is [n_expert_used, n_tokens] while the selection score is
        // [n_expert, n_tokens]. Width disambiguates them.
        if (ne != st->cand_ne[il]) return true;

        st->last_scores[il] = std::move(V);
        st->last_ne[il] = ne;
        st->last_nt[il] = nt;
        try_compare(st, il);
        return true;
    }
    if (!is_topk) {
        return true;
    }

    // ffn_moe_topk holds the selected expert ids: I32, [n_expert_used, n_tokens].
    if (t->type != GGML_TYPE_I32) {
        return true;
    }

    // ffn_moe_topk is a NON-CONTIGUOUS VIEW: ggml_argsort_top_k sorts all
    // n_expert entries and then views the first n_expert_used columns, so the row
    // stride is n_expert*4 bytes, not n_expert_used*4. Reading it flat (as this
    // did until 2026-07-30) walks along ONE token's full ranking instead of across
    // tokens - i.e. it counted token 0's top-16 rather than each token's top-4.
    // Always index through nb[]; never assume packing on a tensor you did not
    // allocate.
    const size_t row_stride = t->nb[1];
    const int    k_used     = (int) t->ne[0];
    const int    n_tok      = (int) (t->ne[1] * t->ne[2] * t->ne[3]);
    const size_t span       = row_stride * (size_t) (n_tok - 1) + (size_t) k_used * t->nb[0];
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    const uint8_t * base;
    if (is_host) {
        base = (const uint8_t *) t->data;
    } else {
        st->scratch.resize(span);
        ggml_backend_tensor_get(t, st->scratch.data(), 0, span);
        base = st->scratch.data();
    }

    // (n unused: everything below indexes through nb[] rather than assuming packing)
    const int il = layer_of(t->name);
    auto & hist = st->per_layer[il];

    // stash the selection; compare from whichever side arrives second
    if (!st->hash_layers.count(il)) {
        auto & buf = st->last_topk[il];
        buf.resize((size_t) k_used * n_tok);
        for (int tk = 0; tk < n_tok; ++tk) {
            const int32_t * row = (const int32_t *) (base + (size_t) tk * row_stride);
            for (int j = 0; j < k_used; ++j) buf[(size_t) tk * k_used + j] = row[j];
        }
        st->last_kk[il] = k_used;
        st->last_topk_nt[il] = n_tok;
        if (st->chosen_score.find(il) == st->chosen_score.end()) try_elect(st, il);
        else try_compare(st, il);
    }

    for (int tk = 0; tk < n_tok; ++tk) {
        const int32_t * row = (const int32_t *) (base + (size_t) tk * row_stride);
        for (int j = 0; j < k_used; ++j) {
            const int32_t e = row[j];
            if (e < 0) continue;
            if ((size_t) e >= hist.size()) {
                hist.resize(e + 1, 0);
            }
            hist[e]++;
            st->total_selections++;
            st->n_expert_guess = std::max(st->n_expert_guess, e + 1);
        }
    }
    if (st->dump_topk) {
        auto & sq = st->topk_seq[il];
        st->topk_k[il] = k_used;
        for (int tk = 0; tk < n_tok; ++tk) {
            const int32_t * row = (const int32_t *) (base + (size_t) tk * row_stride);
            sq.insert(sq.end(), row, row + k_used);
        }
    }
    return true;
}

// Experts needed to cover `frac` of one layer's selections, sorted descending.
static int experts_for(const std::vector<int64_t> & hist, double frac) {
    std::vector<int64_t> v(hist);
    std::sort(v.begin(), v.end(), std::greater<int64_t>());
    const int64_t total = std::accumulate(v.begin(), v.end(), (int64_t) 0);
    if (total == 0) return 0;
    int64_t acc = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        acc += v[i];
        if ((double) acc / (double) total >= frac) return (int) i + 1;
    }
    return (int) v.size();
}

static double median_of(std::vector<int> xs) {
    if (xs.empty()) return 0;
    std::sort(xs.begin(), xs.end());
    return xs[xs.size() / 2];
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_predict = 0;   // prompt processing only; that is where routing happens

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    route_stats st;
    st.collect_gate = getenv("MOE_STATS_COLLECT_GATE") != nullptr;
    params.cb_eval = cb_routes;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    // This fork returns a unique_ptr and exposes model()/context() as methods.
    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model   * model = llama_init ? llama_init->model()   : nullptr;
    llama_context * ctx   = llama_init ? llama_init->context() : nullptr;
    if (!model || !ctx) {
        LOG_ERR("%s: failed to load model\n", __func__);
        return 1;
    }

    // One prompt per line from --file, else the single --prompt.
    std::vector<std::string> prompts;
    if (!params.prompt.empty()) {
        std::istringstream iss(params.prompt);
        std::string line;
        while (std::getline(iss, line)) {
            if (!line.empty()) prompts.push_back(line);
        }
    }
    if (prompts.empty()) {
        LOG_ERR("%s: no prompts; pass -f prompts.txt or -p \"...\"\n", __func__);
        return 1;
    }

    st.dump_topk = getenv("MOE_TOPK_DUMP") != nullptr;
    LOG_INF("%s: %zu prompts\n", __func__, prompts.size());
    for (size_t i = 0; i < prompts.size(); ++i) {
        std::vector<llama_token> tok = common_tokenize(ctx, prompts[i], true, true);
        if (tok.empty()) continue;
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch batch = llama_batch_get_one(tok.data(), (int32_t) tok.size());
        if (llama_decode(ctx, batch)) {
            LOG_ERR("%s: decode failed on prompt %zu\n", __func__, i);
            continue;
        }
        if ((i + 1) % 10 == 0) {
            LOG_INF("%s:   %zu/%zu prompts, %lld selections\n",
                    __func__, i + 1, prompts.size(), (long long) st.total_selections);
        }
        if (st.dump_topk && !st.topk_seq.empty()) {
            const auto & first = *st.topk_seq.begin();
            const int kk = st.topk_k.begin()->second;
            if (kk > 0) st.prompt_bounds.push_back((int64_t) first.second.size() / kk);
        }
    }

    // ---- report ----------------------------------------------------------
    std::vector<int> e50, e80, e90, e95;
    for (auto & [il, hist] : st.per_layer) {
        e50.push_back(experts_for(hist, 0.50));
        e80.push_back(experts_for(hist, 0.80));
        e90.push_back(experts_for(hist, 0.90));
        e95.push_back(experts_for(hist, 0.95));
    }

    const int n_expert = st.n_expert_guess;
    int64_t selfcheck_tokens = 0;
    int64_t selfcheck_mismatch = 0;
    int64_t selfcheck_tie_equivalent = 0;
    for (auto & [il, v] : st.check_tokens) {
        (void) il;
        selfcheck_tokens += v;
    }
    for (auto & [il, v] : st.check_mismatch) {
        (void) il;
        selfcheck_mismatch += v;
    }
    for (auto & [il, v] : st.check_tie_equivalent) {
        (void) il;
        selfcheck_tie_equivalent += v;
    }
    const bool selfcheck_passed = selfcheck_tokens > 0 && selfcheck_mismatch == 0;
    LOG("\n");
    LOG("=== MoE route concentration ===\n");
    LOG("  layers observed      : %zu\n", st.per_layer.size());
    LOG("  experts seen         : %d\n", n_expert);
    LOG("  total selections     : %lld\n", (long long) st.total_selections);
    LOG("\n  experts needed per layer to cover, median across layers:\n");
    LOG("    50%% of routes : %5.0f  (%.1f%% of experts)\n",
        median_of(e50), 100.0 * median_of(e50) / std::max(1, n_expert));
    LOG("    80%% of routes : %5.0f  (%.1f%% of experts)\n",
        median_of(e80), 100.0 * median_of(e80) / std::max(1, n_expert));
    LOG("    90%% of routes : %5.0f  (%.1f%% of experts)\n",
        median_of(e90), 100.0 * median_of(e90) / std::max(1, n_expert));
    LOG("    95%% of routes : %5.0f  (%.1f%% of experts)\n",
        median_of(e95), 100.0 * median_of(e95) / std::max(1, n_expert));
    LOG("\n  MoE tensors observed by the callback:\n");
    for (auto & [nm, c] : st.seen_names) LOG("    %-28s %d\n", nm.c_str(), c);
    LOG("\n  Uniform routing would need %.0f%%/%.0f%%/%.0f%% of experts for 50/80/90%%.\n",
        50.0, 80.0, 90.0);
    LOG("  Concentration is the gap between those and the measured rows above.\n");

    const char * env_out = getenv("MOE_STATS_OUT");
    const std::string out_path = env_out ? env_out
                               : (params.out_file.empty() ? std::string() : params.out_file);
    if (!out_path.empty()) {
        std::ofstream f(out_path);
        f << "{\n  \"n_expert\": " << n_expert
          << ",\n  \"layers\": " << st.per_layer.size()
          << ",\n  \"total_selections\": " << st.total_selections
          << ",\n  \"self_check\": {\"tokens\": " << selfcheck_tokens
          << ", \"matches\": " << (selfcheck_tokens - selfcheck_mismatch)
          << ", \"mismatches\": " << selfcheck_mismatch
          << ", \"cutoff_tie_equivalent\": " << selfcheck_tie_equivalent
          << ", \"exact_id_sets\": "
          << (selfcheck_tokens - selfcheck_mismatch - selfcheck_tie_equivalent)
          << ", \"passed\": " << (selfcheck_passed ? "true" : "false") << "}"
          << ",\n  \"routing_kind\": {";
        bool fr = true;
        for (auto & [il, hist] : st.per_layer) {
            (void) hist;
            if (!fr) f << ",";
            fr = false;
            f << "\"" << il << "\":\""
              << (st.hash_layers.count(il) ? "token_hash" : "learned_score") << "\"";
        }
        f << "}"
          << ",\n  \"median_experts_for\": {\"p50\": " << median_of(e50)
          << ", \"p80\": " << median_of(e80)
          << ", \"p90\": " << median_of(e90)
          << ", \"p95\": " << median_of(e95) << "},\n";
        f << "  \"per_layer\": {\n";
        bool first = true;
        for (auto & [il, hist] : st.per_layer) {
            if (!first) f << ",\n";
            first = false;
            f << "    \"" << il << "\": [";
            for (size_t i = 0; i < hist.size(); ++i) {
                if (i) f << ",";
                f << hist[i];
            }
            f << "]";
        }
        f << "\n  },\n";
        // Router-logit first and second moments, per layer. These are what the
        // reachability analysis consumes; the selection histogram alone cannot
        // distinguish "never fired" from "never sampled".
        f << "  \"logit_moments\": {\n";
        bool f2 = true;
        for (auto & [il, mu] : st.logit_sum) {
            if (!f2) f << ",\n";
            f2 = false;
            const int64_t n = st.logit_tokens[il];
            const auto & sq = st.logit_sumsq[il];
            const int ne = (int) mu.size();
            f << "    \"" << il << "\": {\"tokens\": " << n << ", \"mean\": [";
            for (int i = 0; i < ne; ++i) { if (i) f << ","; f << (n ? mu[i] / n : 0.0); }
            f << "], \"cov_upper\": [";
            bool f3 = true;
            for (int i = 0; i < ne; ++i) {
                for (int j = i; j < ne; ++j) {
                    if (!f3) f << ",";
                    f3 = false;
                    const double e_ij = n ? sq[(size_t) i * ne + j] / n : 0.0;
                    const double mi = n ? mu[i] / n : 0.0, mj = n ? mu[j] / n : 0.0;
                    f << (e_ij - mi * mj);
                }
            }
            f << "], \"maxdiff\": [";
            const auto & md = st.logit_maxdiff[il];
            for (size_t i = 0; i < md.size(); ++i) { if (i) f << ","; f << md[i]; }
            f << "], \"selfcheck_tokens\": "
              << (st.check_tokens.count(il) ? st.check_tokens.at(il) : 0)
              << ", \"selfcheck_mismatch\": "
              << (st.check_mismatch.count(il) ? st.check_mismatch.at(il) : 0)
              << ", \"selfcheck_cutoff_tie_equivalent\": "
              << (st.check_tie_equivalent.count(il) ? st.check_tie_equivalent.at(il) : 0)
              << ", \"noncontiguous\": " << (st.noncontig.count(il) ? 1 : 0) << "}";
        }
        f << "\n  },\n  \"gate_max\": {\n";
        bool f4 = true;
        for (auto & [il, gm] : st.gate_max) {
            if (!f4) f << ",\n";
            f4 = false;
            f << "    \"" << il << "\": [";
            for (size_t i = 0; i < gm.size(); ++i) { if (i) f << ","; f << gm[i]; }
            f << "]";
        }
        f << "\n  }\n}\n";
        LOG("\n  wrote %s\n", out_path.c_str());
    }

    {
        const int64_t ct = selfcheck_tokens;
        const int64_t cm = selfcheck_mismatch;
        int nc = 0;
        for (auto & [il, v] : st.noncontig) { (void) il; (void) v; nc++; }
        LOG("\n  elected selection score per layer:\n");
        if (!st.hash_layers.empty()) {
            LOG("    token-hash layers (traffic only; score reachability N/A):");
            for (auto & [il, v] : st.hash_layers) { (void) v; LOG(" %d", il); }
            LOG("\n");
        }
        for (auto & [il, nm] : st.chosen_score) {
            LOG("    layer %-3d %-24s (reproduced %.1f%% of the election ubatch)\n",
                il, nm.c_str(), 100.0 * st.elect_agree[il]);
        }
        LOG("\n  SELF-CHECK: %lld of %lld token-selections valid from the score "
            "tensor we accumulated (%lld exact-id, %lld cutoff-tie equivalent)%s\n",
            (long long) (ct - cm), (long long) ct,
            (long long) (ct - cm - selfcheck_tie_equivalent),
            (long long) selfcheck_tie_equivalent,
            nc ? " (WARNING: non-contiguous score tensor seen)" : "");
        if (ct == 0) {
            LOG("  no comparison was possible - the score and topk tensors never "
                "paired up. Treat every reachability number as unvalidated.\n");
        } else if (cm) {
            LOG("  MISMATCH on %.2f%% of tokens. The tensor we accumulate is NOT the\n"
                "  score the router compared, so hull/ellipsoid reachability results\n"
                "  from this run are INVALID. Counting results are unaffected: they\n"
                "  come from ffn_moe_topk directly.\n", 100.0 * (double) cm / (double) ct);
        } else {
            LOG("  ordering agreement - the reachability results rest on the score\n"
                "  the router compared; exact cutoff ties are reported separately.\n");
        }
    }

    if (st.dump_topk) {
        const char * dp = getenv("MOE_TOPK_DUMP");
        std::ofstream f(dp);
        f << "{\n  \"prompt_bounds\": [";
        for (size_t i = 0; i < st.prompt_bounds.size(); ++i) {
            if (i) f << ",";
            f << st.prompt_bounds[i];
        }
        f << "],\n  \"k\": {";
        bool fk = true;
        for (auto & [il, kk] : st.topk_k) {
            if (!fk) f << ", ";
            fk = false;
            f << "\"" << il << "\": " << kk;
        }
        f << "},\n  \"seq\": {\n";
        bool fl = true;
        for (auto & [il, sq] : st.topk_seq) {
            if (!fl) f << ",\n";
            fl = false;
            f << "    \"" << il << "\": [";
            for (size_t i = 0; i < sq.size(); ++i) {
                if (i) f << ",";
                f << sq[i];
            }
            f << "]";
        }
        f << "\n  }\n}\n";
        LOG("\n  wrote per-token top-k sequences to %s\n", dp);
    }

    llama_backend_free();
    if (!selfcheck_passed) {
        LOG_ERR("reachability output is invalid: score/top-k self-check did not pass\n");
        return 2;
    }
    return 0;
}
