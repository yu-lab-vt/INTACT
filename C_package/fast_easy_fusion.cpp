#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

extern "C" {
    void edt_3d(const unsigned char* ref_cell,
                const int* s1_dims,
                int s1_ndims,
                const unsigned char* mov_cell,
                const int* s2_dims,
                int s2_ndims,
                const float* shift,
                float* output);

    typedef long long price_t;
    price_t* pyCS2(long* msz,
                   double* mtail,
                   double* mhead,
                   double* mlow,
                   double* macap,
                   double* mcost);

    void pyFreeTrackVec(price_t* track_vec);
}

struct DSU {
    std::vector<int> parent;
    std::vector<unsigned char> rank;
    explicit DSU(int n) : parent(n), rank(n, 0) {
        std::iota(parent.begin(), parent.end(), 0);
    }
    int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }
    void unite(int a, int b) {
        int ra = find(a);
        int rb = find(b);
        if (ra == rb) return;
        if (rank[ra] < rank[rb]) parent[ra] = rb;
        else if (rank[ra] > rank[rb]) parent[rb] = ra;
        else { parent[rb] = ra; rank[ra]++; }
    }
};

struct DetArc {
    double id;
    double c_en;
    double c_ex;
    double c_det;
};

struct TransArc {
    double src;
    double dst;
    double cost;
};

struct CostJob {
    size_t arc_pos;
    int u;
    int v;
};

static inline int64_t c_index_2d(int64_t y, int64_t x, int64_t X) {
    return y * X + x;
}

static inline int64_t c_index_3d(int64_t z, int64_t y, int64_t x, int64_t Y, int64_t X) {
    return (z * Y + y) * X + x;
}

static void build_subgraphs_internal(
    const py::detail::unchecked_reference<int64_t, 1>& frames1,
    const py::detail::unchecked_reference<int64_t, 1>& parents1,
    const py::detail::unchecked_reference<int64_t, 1>& frames2,
    const py::detail::unchecked_reference<int64_t, 1>& parents2,
    const py::detail::unchecked_reference<int64_t, 2>& matches,
    std::vector<std::vector<int>>& easy_subgraphs,
    std::vector<std::vector<int>>& hard_subgraphs,
    std::vector<std::vector<int>>& dir_edges
) {
    const int n1 = static_cast<int>(frames1.shape(0));
    const int n2 = static_cast<int>(frames2.shape(0));
    const int N = n1 + n2;
    auto m2 = [n1](int j) { return n1 + j; };

    DSU dsu(N);
    dir_edges.assign(N, {});
    std::vector<std::vector<int>> children1(n1), children2(n2);

    for (int i = 0; i < n1; ++i) {
        int p = static_cast<int>(parents1(i));
        if (p != -1) {
            if (p < 0 || p >= n1) throw std::runtime_error("Invalid parent index in parents1.");
            children1[p].push_back(i);
            dir_edges[p].push_back(i);
            dsu.unite(p, i);
        }
    }

    for (int j = 0; j < n2; ++j) {
        int p = static_cast<int>(parents2(j));
        if (p != -1) {
            if (p < 0 || p >= n2) throw std::runtime_error("Invalid parent index in parents2.");
            children2[p].push_back(j);
            int src = m2(p);
            int dst = m2(j);
            dir_edges[src].push_back(dst);
            dsu.unite(src, dst);
        }
    }

    const int M = static_cast<int>(matches.shape(0));
    for (int k = 0; k < M; ++k) {
        int i1 = static_cast<int>(matches(k, 0));
        int i2 = static_cast<int>(matches(k, 1));
        if (i1 < 0 || i1 >= n1 || i2 < 0 || i2 >= n2) {
            throw std::runtime_error("Invalid match index.");
        }

        int u = i1;
        int v = m2(i2);
        dsu.unite(u, v);

        int p1 = static_cast<int>(parents1(i1));
        int p2 = static_cast<int>(parents2(i2));

        if (p1 != -1) {
            dsu.unite(v, p1);
            dir_edges[p1].push_back(v);
        }
        if (p2 != -1) {
            int p2g = m2(p2);
            dsu.unite(u, p2g);
            dir_edges[p2g].push_back(u);
        }
        for (int c1 : children1[i1]) {
            dsu.unite(v, c1);
            dir_edges[v].push_back(c1);
        }
        for (int c2 : children2[i2]) {
            int c2g = m2(c2);
            dsu.unite(u, c2g);
            dir_edges[u].push_back(c2g);
        }
    }

    for (auto& nbrs : dir_edges) {
        if (nbrs.size() > 1) {
            std::sort(nbrs.begin(), nbrs.end());
            nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
        }
    }

    std::vector<int> root_to_comp(N, -1);
    std::vector<std::vector<int>> comps;
    comps.reserve(N);
    for (int u = 0; u < N; ++u) {
        int r = dsu.find(u);
        if (root_to_comp[r] < 0) {
            root_to_comp[r] = static_cast<int>(comps.size());
            comps.emplace_back();
        }
        comps[root_to_comp[r]].push_back(u);
    }

    int64_t min_f1 = std::numeric_limits<int64_t>::max(), max_f1 = std::numeric_limits<int64_t>::min();
    int64_t min_f2 = std::numeric_limits<int64_t>::max(), max_f2 = std::numeric_limits<int64_t>::min();
    for (int i = 0; i < n1; ++i) { min_f1 = std::min(min_f1, frames1(i)); max_f1 = std::max(max_f1, frames1(i)); }
    for (int j = 0; j < n2; ++j) { min_f2 = std::min(min_f2, frames2(j)); max_f2 = std::max(max_f2, frames2(j)); }
    int64_t off1 = min_f1, off2 = min_f2;
    size_t len1 = static_cast<size_t>(max_f1 - min_f1 + 1);
    size_t len2 = static_cast<size_t>(max_f2 - min_f2 + 1);
    std::vector<int> seen1(len1, -1), seen2(len2, -1);

    int stamp = 0;
    for (const auto& nodes : comps) {
        ++stamp;
        bool easy = true;
        for (int u : nodes) {
            if (u < n1) {
                size_t idx = static_cast<size_t>(frames1(u) - off1);
                if (seen1[idx] == stamp) { easy = false; break; }
                seen1[idx] = stamp;
            } else {
                int j = u - n1;
                size_t idx = static_cast<size_t>(frames2(j) - off2);
                if (seen2[idx] == stamp) { easy = false; break; }
                seen2[idx] = stamp;
            }
        }
        if (easy) easy_subgraphs.push_back(nodes);
        else hard_subgraphs.push_back(nodes);
    }
}

static double overlap_cost_2d(
    int u,
    int v,
    const int32_t* vox,
    const int64_t* offsets
) {
    int64_t su = offsets[u];
    int64_t eu = offsets[u + 1];
    int64_t sv = offsets[v];
    int64_t ev = offsets[v + 1];

    auto key2 = [](int32_t a, int32_t b) -> int64_t {
        return (static_cast<int64_t>(a) << 32) ^ static_cast<uint32_t>(b);
    };

    std::unordered_set<int64_t> set_u;
    std::unordered_set<int64_t> set_v;

    set_u.reserve(static_cast<size_t>(eu - su) * 2 + 1);
    set_v.reserve(static_cast<size_t>(ev - sv) * 2 + 1);

    for (int64_t i = su; i < eu; ++i) {
        set_u.insert(key2(vox[2 * i], vox[2 * i + 1]));
    }

    for (int64_t i = sv; i < ev; ++i) {
        set_v.insert(key2(vox[2 * i], vox[2 * i + 1]));
    }

    int64_t inter = 0;
    if (set_u.size() <= set_v.size()) {
        for (const auto& key : set_u) {
            if (set_v.find(key) != set_v.end()) {
                ++inter;
            }
        }
    } else {
        for (const auto& key : set_v) {
            if (set_u.find(key) != set_u.end()) {
                ++inter;
            }
        }
    }

    const double uni = static_cast<double>(set_u.size() + set_v.size() - inter);
    const double ratio = (uni > 0.0) ? static_cast<double>(inter) / uni : 0.0;

    return ratio > 0.0 ? -std::log(ratio) : 1e4;
}

static double overlap_cost_3d(int u, int v,
                              const int32_t* vox,
                              const int64_t* offsets) {
    int64_t su = offsets[u], eu = offsets[u + 1];
    int64_t sv = offsets[v], ev = offsets[v + 1];
    int64_t nu = eu - su, nv = ev - sv;
    if (nu < 2 || nv < 2) return 100.0;

    int32_t min1[3] = {std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::max()};
    int32_t max1[3] = {std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::min()};
    int32_t min2[3] = {std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::max()};
    int32_t max2[3] = {std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::min()};

    for (int64_t i = su; i < eu; ++i) {
        for (int d = 0; d < 3; ++d) {
            int32_t val = vox[3 * i + d];
            min1[d] = std::min(min1[d], val);
            max1[d] = std::max(max1[d], val);
        }
    }
    for (int64_t i = sv; i < ev; ++i) {
        for (int d = 0; d < 3; ++d) {
            int32_t val = vox[3 * i + d];
            min2[d] = std::min(min2[d], val);
            max2[d] = std::max(max2[d], val);
        }
    }

    int dims1[3] = {max1[0] - min1[0] + 1, max1[1] - min1[1] + 1, max1[2] - min1[2] + 1};
    int dims2[3] = {max2[0] - min2[0] + 1, max2[1] - min2[1] + 1, max2[2] - min2[2] + 1};

    int64_t sz1 = static_cast<int64_t>(dims1[0]) * dims1[1] * dims1[2];
    int64_t sz2 = static_cast<int64_t>(dims2[0]) * dims2[1] * dims2[2];
    if (sz1 <= 0 || sz2 <= 0) return 100.0;

    std::vector<unsigned char> mask1(static_cast<size_t>(sz1), 0);
    std::vector<unsigned char> mask2(static_cast<size_t>(sz2), 0);
    std::vector<int32_t> rel1(static_cast<size_t>(nu) * 3);
    std::vector<int32_t> rel2(static_cast<size_t>(nv) * 3);

    for (int64_t k = 0, i = su; i < eu; ++i, ++k) {
        int32_t z = vox[3 * i]     - min1[0];
        int32_t y = vox[3 * i + 1] - min1[1];
        int32_t x = vox[3 * i + 2] - min1[2];
        rel1[3 * k] = z; rel1[3 * k + 1] = y; rel1[3 * k + 2] = x;
        mask1[static_cast<size_t>(c_index_3d(z, y, x, dims1[1], dims1[2]))] = 1;
    }
    for (int64_t k = 0, i = sv; i < ev; ++i, ++k) {
        int32_t z = vox[3 * i]     - min2[0];
        int32_t y = vox[3 * i + 1] - min2[1];
        int32_t x = vox[3 * i + 2] - min2[2];
        rel2[3 * k] = z; rel2[3 * k + 1] = y; rel2[3 * k + 2] = x;
        mask2[static_cast<size_t>(c_index_3d(z, y, x, dims2[1], dims2[2]))] = 1;
    }

    float shift12[3] = {
        static_cast<float>(min2[0] - min1[0]),
        static_cast<float>(min2[1] - min1[1]),
        static_cast<float>(min2[2] - min1[2])
    };
    float shift21[3] = {-shift12[0], -shift12[1], -shift12[2]};

    std::vector<float> out12(static_cast<size_t>(sz2), 0.0f);
    std::vector<float> out21(static_cast<size_t>(sz1), 0.0f);

    edt_3d(mask1.data(), dims1, 3, mask2.data(), dims2, 3, shift12, out12.data());
    edt_3d(mask2.data(), dims2, 3, mask1.data(), dims1, 3, shift21, out21.data());

    double sum_n2c = 0.0;
    for (int64_t k = 0; k < nv; ++k) {
        int32_t z = rel2[3 * k], y = rel2[3 * k + 1], x = rel2[3 * k + 2];
        float d2 = out12[static_cast<size_t>(c_index_3d(z, y, x, dims2[1], dims2[2]))];
        sum_n2c += std::sqrt(static_cast<double>(std::max(d2, 0.0f)));
    }

    double sum_c2n = 0.0;
    for (int64_t k = 0; k < nu; ++k) {
        int32_t z = rel1[3 * k], y = rel1[3 * k + 1], x = rel1[3 * k + 2];
        float d2 = out21[static_cast<size_t>(c_index_3d(z, y, x, dims1[1], dims1[2]))];
        sum_c2n += std::sqrt(static_cast<double>(std::max(d2, 0.0f)));
    }

    double mean_c2n = sum_c2n / static_cast<double>(nu);
    double mean_n2c = sum_n2c / static_cast<double>(nv);
    return std::max(mean_c2n, mean_n2c);
}

static double edge_cost_cpp(int u, int v,
                            const int32_t* vox,
                            const int64_t* offsets,
                            int dim) {
    if (dim == 3) return overlap_cost_3d(u, v, vox, offsets);
    if (dim == 2) return overlap_cost_2d(u, v, vox, offsets);
    throw std::runtime_error("Only 2D or 3D voxels are supported.");
}

static void build_easy_arcs(
    const std::vector<std::vector<int>>& easy_subgraphs,
    const std::vector<std::vector<int>>& edges,
    const py::detail::unchecked_reference<int64_t, 1>& frames1,
    const py::detail::unchecked_reference<int64_t, 1>& frames2,
    const int32_t* vox,
    const int64_t* offsets,
    int dim,
    double threshold,
    std::vector<DetArc>& detection_arcs,
    std::vector<TransArc>& transition_arcs
) {
    const int n1 = static_cast<int>(frames1.shape(0));
    const int n2 = static_cast<int>(frames2.shape(0));
    const int N = n1 + n2;

    if (N <= 0) {
        detection_arcs.clear();
        transition_arcs.clear();
        return;
    }

    /*
     * This follows Python pruning(...):
     *
     * detection_arcs_original =
     *   [big, 2 * threshold, -2 * threshold - 0.00001]
     *
     * detection_arcs_control =
     *   [threshold, threshold, -2 * threshold - 0.00001]
     *
     * big =
     *   (-(max(frames2) - min(frames1) + 1)
     *      * (-2 * threshold - 0.00001)) + 1
     */
    int64_t min_frame1 = frames1(0);
    int64_t max_frame2 = frames2(0);

    for (int i = 0; i < n1; ++i) {
        min_frame1 = std::min(min_frame1, frames1(i));
    }
    for (int j = 0; j < n2; ++j) {
        max_frame2 = std::max(max_frame2, frames2(j));
    }

    const double threshold2 = -2.0 * threshold - 0.00001;
    const double big =
        (-(static_cast<double>(max_frame2 - min_frame1 + 1)) * threshold2) + 1.0;

    /*
     * Build detection arcs.
     * Node ids must be exactly 0, 1, ..., N + num_easy_subgraphs - 1.
     */
    detection_arcs.clear();
    detection_arcs.resize(static_cast<size_t>(N + easy_subgraphs.size()));

    for (int u = 0; u < N; ++u) {
        detection_arcs[static_cast<size_t>(u)] =
            DetArc{
                static_cast<double>(u),
                big,
                2.0 * threshold,
                threshold2
            };
    }

    for (size_t gi = 0; gi < easy_subgraphs.size(); ++gi) {
        int ctrl = N + static_cast<int>(gi);
        detection_arcs[static_cast<size_t>(ctrl)] =
            DetArc{
                static_cast<double>(ctrl),
                threshold,
                threshold,
                threshold2
            };
    }

    /*
     * We must preserve transition_arcs order.
     *
     * Python order in pruning(...):
     *   for each subgraph i:
     *       for u in subgraph:
     *           for v in edges[u]:
     *               transition_arcs.append([u, v, cost])
     *       for first_frame_node in first_frame_nodes:
     *           transition_arcs.append([N+i, first_frame_node, 0.0])
     *
     * Therefore:
     *   - transition_arcs is filled in this exact structure.
     *   - cost is computed later in parallel by writing back to arc_pos.
     */
    transition_arcs.clear();

    size_t approx_edges = 0;
    for (const auto& sg : easy_subgraphs) {
        for (int u : sg) {
            if (u >= 0 && u < static_cast<int>(edges.size())) {
                approx_edges += edges[static_cast<size_t>(u)].size();
            }
        }
    }

    transition_arcs.reserve(approx_edges + easy_subgraphs.size() * 2);
    std::vector<CostJob> cost_jobs;
    cost_jobs.reserve(approx_edges);

    for (size_t gi = 0; gi < easy_subgraphs.size(); ++gi) {
        const auto& sg = easy_subgraphs[gi];
        if (sg.empty()) {
            continue;
        }

        int64_t minf = std::numeric_limits<int64_t>::max();
        int64_t maxf = std::numeric_limits<int64_t>::min();

        for (int u : sg) {
            int64_t f = (u < n1) ? frames1(u) : frames2(u - n1);
            minf = std::min(minf, f);
            maxf = std::max(maxf, f);
        }

        const int ctrl = N + static_cast<int>(gi);

        /*
         * 1. Real transition arcs.
         * Keep the exact loop order:
         *   for u in subgraph:
         *       for v in edges[u]:
         */
        for (int u : sg) {
            int64_t f = (u < n1) ? frames1(u) : frames2(u - n1);

            if (f == maxf) {
                detection_arcs[static_cast<size_t>(u)].c_ex = threshold;
            }

            if (u < 0 || u >= static_cast<int>(edges.size())) {
                continue;
            }

            const auto& nbrs = edges[static_cast<size_t>(u)];
            for (int v : nbrs) {
                const size_t pos = transition_arcs.size();

                transition_arcs.push_back(
                    TransArc{
                        static_cast<double>(u),
                        static_cast<double>(v),
                        0.0
                    }
                );

                cost_jobs.push_back(
                    CostJob{
                        pos,
                        u,
                        v
                    }
                );
            }
        }

        /*
         * 2. Pseudo-control arcs.
         *
         * Python builds first_frame_nodes by:
         *   min_idx_in_part1 first, then min_idx_in_part2.
         *
         * To mimic this order, write movie1 first-frame nodes first,
         * then movie2 first-frame nodes.
         */
        for (int u : sg) {
            if (u < n1) {
                int64_t f = frames1(u);
                if (f == minf) {
                    transition_arcs.push_back(
                        TransArc{
                            static_cast<double>(ctrl),
                            static_cast<double>(u),
                            0.0
                        }
                    );
                }
            }
        }

        for (int u : sg) {
            if (u >= n1) {
                int j = u - n1;
                int64_t f = frames2(j);
                if (f == minf) {
                    transition_arcs.push_back(
                        TransArc{
                            static_cast<double>(ctrl),
                            static_cast<double>(u),
                            0.0
                        }
                    );
                }
            }
        }
    }

    /*
     * Compute real transition costs in parallel.
     * This does not change transition_arcs order.
     */
    #pragma omp parallel for schedule(dynamic)
    for (long long k = 0; k < static_cast<long long>(cost_jobs.size()); ++k) {
        const CostJob& job = cost_jobs[static_cast<size_t>(k)];

        const double c = edge_cost_cpp(
            job.u,
            job.v,
            vox,
            offsets,
            dim
        );

        transition_arcs[job.arc_pos].cost = c;
    }
}



static std::vector<std::vector<int>> run_cinda_cpp(
    const std::vector<DetArc>& detection_arcs,
    const std::vector<TransArc>& transition_arcs,
    int N_original
) {
    const long n_detection = static_cast<long>(detection_arcs.size());
    const long n_transition = static_cast<long>(transition_arcs.size());
    const long n_arcs = n_detection * 3 + n_transition;

    std::vector<double> mtail(static_cast<size_t>(n_arcs));
    std::vector<double> mhead(static_cast<size_t>(n_arcs));
    std::vector<double> mlow(static_cast<size_t>(n_arcs), 0.0);
    std::vector<double> macap(static_cast<size_t>(n_arcs), 1.0);
    std::vector<double> mcost(static_cast<size_t>(n_arcs));

    long pos = 0;
    for (long i = 0; i < n_detection; ++i, ++pos) {
        double did = detection_arcs[static_cast<size_t>(i)].id + 1.0;
        mtail[static_cast<size_t>(pos)] = 1.0;
        mhead[static_cast<size_t>(pos)] = did * 2.0;
        mcost[static_cast<size_t>(pos)] = detection_arcs[static_cast<size_t>(i)].c_en;
    }
    for (long i = 0; i < n_detection; ++i, ++pos) {
        double did = detection_arcs[static_cast<size_t>(i)].id + 1.0;
        mtail[static_cast<size_t>(pos)] = did * 2.0 + 1.0;
        mhead[static_cast<size_t>(pos)] = 1.0;
        mcost[static_cast<size_t>(pos)] = detection_arcs[static_cast<size_t>(i)].c_ex;
    }
    for (long i = 0; i < n_detection; ++i, ++pos) {
        double did = detection_arcs[static_cast<size_t>(i)].id + 1.0;
        mtail[static_cast<size_t>(pos)] = did * 2.0;
        mhead[static_cast<size_t>(pos)] = did * 2.0 + 1.0;
        mcost[static_cast<size_t>(pos)] = detection_arcs[static_cast<size_t>(i)].c_det;
    }
    for (long i = 0; i < n_transition; ++i, ++pos) {
        double s = transition_arcs[static_cast<size_t>(i)].src + 1.0;
        double t = transition_arcs[static_cast<size_t>(i)].dst + 1.0;
        mtail[static_cast<size_t>(pos)] = s * 2.0 + 1.0;
        mhead[static_cast<size_t>(pos)] = t * 2.0;
        mcost[static_cast<size_t>(pos)] = transition_arcs[static_cast<size_t>(i)].cost;
    }

    // Match the Python mcc4mot behavior: float costs are scaled to integer costs.
    constexpr double SCALE = 1e7;
    for (double& c : mcost) c = static_cast<double>(static_cast<long long>(c * SCALE));

    long msz[3] = {12, 2 * n_detection + 1, n_arcs};

    price_t* track_vec = pyCS2(msz, mtail.data(), mhead.data(), mlow.data(), macap.data(), mcost.data());
    if (track_vec == nullptr) throw std::runtime_error("pyCS2 returned null.");

    long long L = static_cast<long long>(track_vec[0]);
    std::vector<std::vector<int>> tracks;
    std::vector<price_t> sub;
    sub.reserve(128);

    for (long long i = 1; i <= L; ++i) {
        price_t x = track_vec[i];
        if (x > 0) {
            sub.push_back(x);
        } else {
            std::vector<int> tr;
            for (size_t k = 0; k < sub.size(); k += 2) {
                int node = static_cast<int>(sub[k] / 2) - 1;
                if (node >= 0 && node < N_original) tr.push_back(node);
            }
            if (!tr.empty()) tracks.push_back(std::move(tr));
            sub.clear();
        }
    }

    pyFreeTrackVec(track_vec);
    return tracks;
}

py::tuple solve_easy_fusion_cpp(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> frames1_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> parents1_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> frames2_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> parents2_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> matches_arr,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> vox_flat_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> vox_offsets_arr,
    int dim,
    double threshold
) {
    if (matches_arr.ndim() != 2 || matches_arr.shape(1) < 2) throw std::runtime_error("matches must have shape (M, >=2).");
    if (vox_flat_arr.ndim() != 2 || vox_flat_arr.shape(1) != dim) throw std::runtime_error("vox_flat shape does not match dim.");

    auto frames1 = frames1_arr.unchecked<1>();
    auto parents1 = parents1_arr.unchecked<1>();
    auto frames2 = frames2_arr.unchecked<1>();
    auto parents2 = parents2_arr.unchecked<1>();
    auto matches = matches_arr.unchecked<2>();
    auto vox_flat = vox_flat_arr.unchecked<2>();
    auto vox_offsets = vox_offsets_arr.unchecked<1>();

    const int n1 = static_cast<int>(frames1.shape(0));
    const int n2 = static_cast<int>(frames2.shape(0));
    const int N = n1 + n2;
    if (parents1.shape(0) != n1 || parents2.shape(0) != n2) throw std::runtime_error("frames and parents length mismatch.");
    if (vox_offsets.shape(0) != N + 1) throw std::runtime_error("vox_offsets must have length N + 1.");

    std::vector<std::vector<int>> easy_subgraphs;
    std::vector<std::vector<int>> hard_subgraphs;
    std::vector<std::vector<int>> edges;
    build_subgraphs_internal(frames1, parents1, frames2, parents2, matches, easy_subgraphs, hard_subgraphs, edges);

    std::vector<DetArc> detection_arcs;
    std::vector<TransArc> transition_arcs;
    build_easy_arcs(easy_subgraphs, edges, frames1, frames2,
                    vox_flat.data(0, 0), vox_offsets.data(0), dim, threshold,
                    detection_arcs, transition_arcs);

    std::vector<std::vector<int>> tracks_easy = run_cinda_cpp(detection_arcs, transition_arcs, N);
    return py::make_tuple(tracks_easy, hard_subgraphs, edges);
}

py::tuple build_easy_arcs_debug_cpp(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> frames1_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> parents1_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> frames2_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> parents2_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> matches_arr,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> vox_flat_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> vox_offsets_arr,
    int dim,
    double threshold
) {
    if (matches_arr.ndim() != 2 || matches_arr.shape(1) < 2) {
        throw std::runtime_error("matches must have shape (M, >=2).");
    }
    if (vox_flat_arr.ndim() != 2 || vox_flat_arr.shape(1) != dim) {
        throw std::runtime_error("vox_flat shape does not match dim.");
    }

    auto frames1 = frames1_arr.unchecked<1>();
    auto parents1 = parents1_arr.unchecked<1>();
    auto frames2 = frames2_arr.unchecked<1>();
    auto parents2 = parents2_arr.unchecked<1>();
    auto matches = matches_arr.unchecked<2>();
    auto vox_flat = vox_flat_arr.unchecked<2>();
    auto vox_offsets = vox_offsets_arr.unchecked<1>();

    const int n1 = static_cast<int>(frames1.shape(0));
    const int n2 = static_cast<int>(frames2.shape(0));
    const int N = n1 + n2;

    std::vector<std::vector<int>> easy_subgraphs;
    std::vector<std::vector<int>> hard_subgraphs;
    std::vector<std::vector<int>> edges;

    build_subgraphs_internal(
        frames1,
        parents1,
        frames2,
        parents2,
        matches,
        easy_subgraphs,
        hard_subgraphs,
        edges
    );

    std::vector<DetArc> detection_arcs_vec;
    std::vector<TransArc> transition_arcs_vec;

    build_easy_arcs(
        easy_subgraphs,
        edges,
        frames1,
        frames2,
        vox_flat.data(0, 0),
        vox_offsets.data(0),
        dim,
        threshold,
        detection_arcs_vec,
        transition_arcs_vec
    );

    const py::ssize_t det_rows =
        static_cast<py::ssize_t>(detection_arcs_vec.size());
    const py::ssize_t trans_rows =
        static_cast<py::ssize_t>(transition_arcs_vec.size());

    std::vector<py::ssize_t> det_shape = {det_rows, 4};
    std::vector<py::ssize_t> trans_shape = {trans_rows, 3};

    py::array_t<double> detection_arcs(det_shape);
    py::array_t<double> transition_arcs(trans_shape);

    auto det = detection_arcs.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(detection_arcs_vec.size()); ++i) {
        det(i, 0) = detection_arcs_vec[static_cast<size_t>(i)].id;
        det(i, 1) = detection_arcs_vec[static_cast<size_t>(i)].c_en;
        det(i, 2) = detection_arcs_vec[static_cast<size_t>(i)].c_ex;
        det(i, 3) = detection_arcs_vec[static_cast<size_t>(i)].c_det;
    }

    auto tr = transition_arcs.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(transition_arcs_vec.size()); ++i) {
        tr(i, 0) = transition_arcs_vec[static_cast<size_t>(i)].src;
        tr(i, 1) = transition_arcs_vec[static_cast<size_t>(i)].dst;
        tr(i, 2) = transition_arcs_vec[static_cast<size_t>(i)].cost;
    }

    return py::make_tuple(
        detection_arcs,
        transition_arcs,
        easy_subgraphs,
        hard_subgraphs,
        edges
    );
}


PYBIND11_MODULE(fast_easy_fusion, m) {
    m.doc() = "Fast C++ implementation of INTACT Step 4-1 and Step 4-2 easy-subgraph fusion.";

    m.def("solve_easy_fusion_cpp", &solve_easy_fusion_cpp,
          py::arg("frames1"), py::arg("parents1"),
          py::arg("frames2"), py::arg("parents2"),
          py::arg("matches"),
          py::arg("vox_flat"), py::arg("vox_offsets"),
          py::arg("dim"), py::arg("threshold"));

    m.def("build_easy_arcs_debug_cpp", &build_easy_arcs_debug_cpp,
          py::arg("frames1"), py::arg("parents1"),
          py::arg("frames2"), py::arg("parents2"),
          py::arg("matches"),
          py::arg("vox_flat"), py::arg("vox_offsets"),
          py::arg("dim"), py::arg("threshold"));
}
