#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>


#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <chrono>
#include <queue>
#include <string>
#include <unordered_map>
#include <sstream>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;


// ============================================================
// 1. Basic geometry utilities
// ============================================================

struct Point3i {
    int z;
    int y;
    int x;
};

static inline int64_t idx3d(int z, int y, int x, int Y, int X) {
    return (static_cast<int64_t>(z) * Y + y) * X + x;
}

static inline int64_t c_index_3d(
    int64_t z,
    int64_t y,
    int64_t x,
    int64_t Y,
    int64_t X
) {
    return (z * Y + y) * X + x;
}

/*
 * New unified voxel reader.
 *
 * Python new behavior:
 *   2D vox: [y, x] -> pseudo-3D [z=0, y, x]
 *   3D vox: [z, y, x]
 *
 * Important:
 *   vox_flat is packed as movie1 vox + movie2 vox.
 *   Therefore merged global node id can directly index offsets:
 *       movie1 node u      -> cell_id = u
 *       movie2 node n1 + j -> cell_id = n1 + j
 */
static std::vector<Point3i> get_cell_voxels_as_zyx_ptr(
    const int32_t* vox,
    const int64_t* offsets,
    int64_t cell_id,
    int dim
) {
    if (cell_id < 0) {
        throw std::runtime_error("Negative cell_id in get_cell_voxels_as_zyx_ptr.");
    }

    const int64_t start = offsets[cell_id];
    const int64_t end = offsets[cell_id + 1];

    if (end < start) {
        throw std::runtime_error("Invalid voxel offsets: end < start.");
    }

    std::vector<Point3i> pts;
    pts.reserve(static_cast<size_t>(end - start));

    for (int64_t i = start; i < end; ++i) {
        if (dim == 2) {
            /*
             * Python:
             *   original 2D vox is [y, x]
             *   lifted to [z=0, y, x]
             */
            int y = static_cast<int>(vox[2 * i]);
            int x = static_cast<int>(vox[2 * i + 1]);
            pts.push_back(Point3i{0, y, x});
        } else if (dim == 3) {
            /*
             * Python convention:
             *   3D vox is already [z, y, x]
             */
            int z = static_cast<int>(vox[3 * i]);
            int y = static_cast<int>(vox[3 * i + 1]);
            int x = static_cast<int>(vox[3 * i + 2]);
            pts.push_back(Point3i{z, y, x});
        } else {
            throw std::runtime_error("Unsupported voxel dimension. Expected dim=2 or dim=3.");
        }
    }

    return pts;
}


// ============================================================
// 2. External C functions
// ============================================================

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


// ============================================================
// 3. New EDT-based region distance
// ============================================================

/*
 * This implements Python ovDistanceRegion(...)[0] for the default branch:
 *
 *   - 2D has already been lifted to pseudo-3D before entering this function.
 *   - 3D is directly used.
 *   - If either region has fewer than 2 voxels, return 100.0.
 *   - Build local bbox masks.
 *   - Call edt_3d twice:
 *       dist2cell1 = edt_3d(mask1, mask2, mov_shift)
 *       dist2cell2 = edt_3d(mask2, mask1, -mov_shift)
 *   - Take sqrt of squared distances.
 *   - Return max(mean(cur->next), mean(next->cur)).
 */
static double region_distance_edt_max_zyx(
    const std::vector<Point3i>& cur_pts,
    const std::vector<Point3i>& next_pts
) {
    if (cur_pts.size() < 2 || next_pts.size() < 2) {
        return 100.0;
    }

    auto get_bbox = [](const std::vector<Point3i>& pts) {
        std::array<int, 3> mn = {pts[0].z, pts[0].y, pts[0].x};
        std::array<int, 3> mx = {pts[0].z, pts[0].y, pts[0].x};

        for (const auto& p : pts) {
            mn[0] = std::min(mn[0], p.z);
            mn[1] = std::min(mn[1], p.y);
            mn[2] = std::min(mn[2], p.x);

            mx[0] = std::max(mx[0], p.z);
            mx[1] = std::max(mx[1], p.y);
            mx[2] = std::max(mx[2], p.x);
        }

        return std::make_pair(mn, mx);
    };

    auto bbox1 = get_bbox(cur_pts);
    auto bbox2 = get_bbox(next_pts);

    auto st1 = bbox1.first;
    auto ed1 = bbox1.second;
    auto st2 = bbox2.first;
    auto ed2 = bbox2.second;

    int Z1 = ed1[0] - st1[0] + 1;
    int Y1 = ed1[1] - st1[1] + 1;
    int X1 = ed1[2] - st1[2] + 1;

    int Z2 = ed2[0] - st2[0] + 1;
    int Y2 = ed2[1] - st2[1] + 1;
    int X2 = ed2[2] - st2[2] + 1;

    if (Z1 <= 0 || Y1 <= 0 || X1 <= 0 ||
        Z2 <= 0 || Y2 <= 0 || X2 <= 0) {
        throw std::runtime_error("Invalid bounding box size in region_distance_edt_max_zyx.");
    }

    const int64_t sz1 = static_cast<int64_t>(Z1) * Y1 * X1;
    const int64_t sz2 = static_cast<int64_t>(Z2) * Y2 * X2;

    if (sz1 <= 0 || sz2 <= 0) {
        return 100.0;
    }

    std::vector<unsigned char> mask1(static_cast<size_t>(sz1), 0);
    std::vector<unsigned char> mask2(static_cast<size_t>(sz2), 0);

    std::vector<Point3i> cell1_sub;
    std::vector<Point3i> cell2_sub;

    cell1_sub.reserve(cur_pts.size());
    cell2_sub.reserve(next_pts.size());

    for (const auto& p : cur_pts) {
        int z = p.z - st1[0];
        int y = p.y - st1[1];
        int x = p.x - st1[2];

        mask1[static_cast<size_t>(idx3d(z, y, x, Y1, X1))] = 1;
        cell1_sub.push_back(Point3i{z, y, x});
    }

    for (const auto& p : next_pts) {
        int z = p.z - st2[0];
        int y = p.y - st2[1];
        int x = p.x - st2[2];

        mask2[static_cast<size_t>(idx3d(z, y, x, Y2, X2))] = 1;
        cell2_sub.push_back(Point3i{z, y, x});
    }

    /*
     * Python:
     *   mov_shift = st_pt2 - st_pt1 - frame_shift
     *
     * In easy fusion, frame_shift is not provided, so it is [0,0,0].
     */
    float shift12[3] = {
        static_cast<float>(st2[0] - st1[0]),
        static_cast<float>(st2[1] - st1[1]),
        static_cast<float>(st2[2] - st1[2])
    };

    float shift21[3] = {
        -shift12[0],
        -shift12[1],
        -shift12[2]
    };

    int dims1[3] = {Z1, Y1, X1};
    int dims2[3] = {Z2, Y2, X2};

    std::vector<float> dist2cell1(static_cast<size_t>(sz2), 0.0f);
    std::vector<float> dist2cell2(static_cast<size_t>(sz1), 0.0f);

    edt_3d(
        mask1.data(),
        dims1,
        3,
        mask2.data(),
        dims2,
        3,
        shift12,
        dist2cell1.data()
    );

    edt_3d(
        mask2.data(),
        dims2,
        3,
        mask1.data(),
        dims1,
        3,
        shift21,
        dist2cell2.data()
    );

    double sum_next_to_cur = 0.0;
    for (const auto& p : cell2_sub) {
        float d2 = dist2cell1[static_cast<size_t>(idx3d(p.z, p.y, p.x, Y2, X2))];
        sum_next_to_cur += std::sqrt(static_cast<double>(std::max(d2, 0.0f)));
    }

    double sum_cur_to_next = 0.0;
    for (const auto& p : cell1_sub) {
        float d2 = dist2cell2[static_cast<size_t>(idx3d(p.z, p.y, p.x, Y1, X1))];
        sum_cur_to_next += std::sqrt(static_cast<double>(std::max(d2, 0.0f)));
    }

    const double mean_cur_to_next =
        sum_cur_to_next / static_cast<double>(cell1_sub.size());

    const double mean_next_to_cur =
        sum_next_to_cur / static_cast<double>(cell2_sub.size());

    return std::max(mean_cur_to_next, mean_next_to_cur);
}

/*
 * New edge cost.
 *
 * Old behavior:
 *   dim == 2 -> overlap/Jaccard cost
 *   dim == 3 -> EDT cost
 *
 * New Python behavior:
 *   dim == 2 -> lift [y,x] to [z=0,y,x], then EDT
 *   dim == 3 -> EDT
 */
static double edge_cost_cpp(
    int u,
    int v,
    const int32_t* vox,
    const int64_t* offsets,
    int dim
) {
    if (dim != 2 && dim != 3) {
        throw std::runtime_error("Only 2D or 3D voxels are supported.");
    }

    std::vector<Point3i> pts_u = get_cell_voxels_as_zyx_ptr(
        vox,
        offsets,
        static_cast<int64_t>(u),
        dim
    );

    std::vector<Point3i> pts_v = get_cell_voxels_as_zyx_ptr(
        vox,
        offsets,
        static_cast<int64_t>(v),
        dim
    );

    return region_distance_edt_max_zyx(pts_u, pts_v);
}


// ============================================================
// 4. Subgraph construction
// ============================================================

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

        if (rank[ra] < rank[rb]) {
            parent[ra] = rb;
        } else if (rank[ra] > rank[rb]) {
            parent[rb] = ra;
        } else {
            parent[rb] = ra;
            rank[ra]++;
        }
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
// ============================================================
// Hard-subgraph B&B data structures
// ============================================================

struct BaseGraph {
    std::vector<DetArc> detection;
    std::vector<TransArc> transition;

    // flow node id -> original global cell id; -1 means virtual/control node
    std::vector<int> expand_to_original;

    // original global cell id -> flow node id; -1 means not present
    std::vector<int> node_in;
    std::vector<int> node_out;
    std::vector<int> node_single;

    // matches_hard in global node ids
    std::vector<std::array<int, 2>> matches;

    int N_original = 0;
    double threshold = 0.0;
};

struct BBState {
    // -1 = not fixed; 0 = choose u / forbid v; 1 = choose v / forbid u
    std::vector<int8_t> fixed_choice;

    // original global cell id -> forbidden?
    std::vector<uint8_t> forbidden_cell;

    int depth = 0;
};

struct FlowResult {
    std::vector<std::vector<int>> trajectories;
    double cost = std::numeric_limits<double>::infinity();
    std::vector<int> violations;
    bool ok = false;
};

struct CorrectionData {
    std::vector<double> min_switch_cost;
    std::vector<double> correction;
    std::vector<double> cost_diff;
    std::vector<double> u_switch_cost;
    std::vector<double> v_switch_cost;
};

struct ConflictBlock {
    int block_id = -1;
    int frame = 0;
    std::vector<int> left_nodes;
    std::vector<int> right_nodes;
    std::vector<int> match_ids;
    int kind = 0;  // 0: 1-vs-k, 1: k-vs-1, 2: k-vs-m
};

struct PQItem {
    double lb;
    int counter;
    int state_id;

    bool operator<(const PQItem& other) const {
        if (lb != other.lb) return lb > other.lb;  // min-heap
        return counter > other.counter;
    }
};

struct HardResult {
    std::vector<std::vector<int>> tracks;
    py::dict stats;
};
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

    auto m2 = [n1](int j) {
        return n1 + j;
    };

    DSU dsu(N);
    dir_edges.assign(N, {});

    std::vector<std::vector<int>> children1(n1);
    std::vector<std::vector<int>> children2(n2);

    /*
     * Intra-movie edges.
     */
    for (int i = 0; i < n1; ++i) {
        int p = static_cast<int>(parents1(i));
        if (p != -1) {
            if (p < 0 || p >= n1) {
                throw std::runtime_error("Invalid parent index in parents1.");
            }

            children1[p].push_back(i);
            dir_edges[p].push_back(i);
            dsu.unite(p, i);
        }
    }

    for (int j = 0; j < n2; ++j) {
        int p = static_cast<int>(parents2(j));
        if (p != -1) {
            if (p < 0 || p >= n2) {
                throw std::runtime_error("Invalid parent index in parents2.");
            }

            children2[p].push_back(j);

            int src = m2(p);
            int dst = m2(j);

            dir_edges[src].push_back(dst);
            dsu.unite(src, dst);
        }
    }

    /*
     * Cross-movie edges induced by matches.
     *
     * This matches Python build_subgraphs:
     *   - u and v are connected for undirected connectivity.
     *   - parent/child cross edges are also added.
     *   - direct match edge itself is not added into dir_edges.
     */
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

    /*
     * Remove duplicate directed edges.
     */
    for (auto& nbrs : dir_edges) {
        if (nbrs.size() > 1) {
            std::sort(nbrs.begin(), nbrs.end());
            nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
        }
    }

    /*
     * Extract connected components from DSU.
     */
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

    /*
     * Easy / hard classification:
     * A component is hard if one movie has more than one node in the same frame.
     * movie1 and movie2 are counted separately.
     */
    int64_t min_f1 = std::numeric_limits<int64_t>::max();
    int64_t max_f1 = std::numeric_limits<int64_t>::min();
    int64_t min_f2 = std::numeric_limits<int64_t>::max();
    int64_t max_f2 = std::numeric_limits<int64_t>::min();

    for (int i = 0; i < n1; ++i) {
        min_f1 = std::min(min_f1, frames1(i));
        max_f1 = std::max(max_f1, frames1(i));
    }

    for (int j = 0; j < n2; ++j) {
        min_f2 = std::min(min_f2, frames2(j));
        max_f2 = std::max(max_f2, frames2(j));
    }

    int64_t off1 = min_f1;
    int64_t off2 = min_f2;

    size_t len1 = static_cast<size_t>(max_f1 - min_f1 + 1);
    size_t len2 = static_cast<size_t>(max_f2 - min_f2 + 1);

    std::vector<int> seen1(len1, -1);
    std::vector<int> seen2(len2, -1);

    int stamp = 0;

    for (const auto& nodes : comps) {
        ++stamp;
        bool easy = true;

        for (int u : nodes) {
            if (u < n1) {
                size_t idx = static_cast<size_t>(frames1(u) - off1);

                if (seen1[idx] == stamp) {
                    easy = false;
                    break;
                }

                seen1[idx] = stamp;
            } else {
                int j = u - n1;
                size_t idx = static_cast<size_t>(frames2(j) - off2);

                if (seen2[idx] == stamp) {
                    easy = false;
                    break;
                }

                seen2[idx] = stamp;
            }
        }

        if (easy) {
            easy_subgraphs.push_back(nodes);
        } else {
            hard_subgraphs.push_back(nodes);
        }
    }
}


// ============================================================
// 5. Easy graph arcs
// ============================================================

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

    detection_arcs.clear();
    detection_arcs.resize(static_cast<size_t>(N + easy_subgraphs.size()));

    /*
     * Original nodes.
     */
    for (int u = 0; u < N; ++u) {
        detection_arcs[static_cast<size_t>(u)] =
            DetArc{
                static_cast<double>(u),
                big,
                2.0 * threshold,
                threshold2
            };
    }

    /*
     * One control node per easy subgraph.
     */
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
     * Reserve transition arcs.
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

    /*
     * Build arcs in the same order as Python pruning.
     */
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
         * Real transition arcs.
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
         * Control arcs.
         *
         * Python order:
         *   first_frame_nodes = concat([min_idx_in_part1, min_idx_in_part2])
         * Therefore movie1 first-frame nodes first, then movie2 first-frame nodes.
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
     * Compute transition costs.
     *
     * The order of transition_arcs is preserved because each job writes back
     * to its original arc_pos.
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


// ============================================================
// 6. CINDA / CS2 solver wrapper
// ============================================================

static std::vector<std::vector<int>> run_cinda_cpp_with_cost(
    const std::vector<DetArc>& detection_arcs,
    const std::vector<TransArc>& transition_arcs,
    int N_original,
    double& total_cost
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

    // 必须和你现在 Python mcc4mot 对齐：np.rint(mcost * 1e7)
    constexpr double SCALE = 1e7;
    for (double& c : mcost) {
        c = static_cast<double>(std::llround(c * SCALE));
    }

    long msz[3] = {12, 2 * n_detection + 1, n_arcs};

    price_t* track_vec = pyCS2(
        msz,
        mtail.data(),
        mhead.data(),
        mlow.data(),
        macap.data(),
        mcost.data()
    );

    if (track_vec == nullptr) {
        throw std::runtime_error("pyCS2 returned null.");
    }

    const long long L = static_cast<long long>(track_vec[0]);

    std::vector<std::vector<int>> tracks;
    std::vector<price_t> sub;
    sub.reserve(128);

    total_cost = 0.0;

    for (long long i = 1; i <= L; ++i) {
        price_t x = track_vec[i];

        if (x > 0) {
            sub.push_back(x);
        } else {
            total_cost += static_cast<double>(x) / SCALE;

            std::vector<int> tr;
            for (size_t k = 0; k < sub.size(); k += 2) {
                int node = static_cast<int>(sub[k] / 2) - 1;
                if (node >= 0 && node < N_original) {
                    tr.push_back(node);
                }
            }

            if (!tr.empty()) {
                tracks.push_back(std::move(tr));
            }

            sub.clear();
        }
    }

    pyFreeTrackVec(track_vec);
    return tracks;
}

static std::vector<std::vector<int>> run_cinda_cpp(
    const std::vector<DetArc>& detection_arcs,
    const std::vector<TransArc>& transition_arcs,
    int N_original
) {
    double total_cost = 0.0;
    return run_cinda_cpp_with_cost(
        detection_arcs,
        transition_arcs,
        N_original,
        total_cost
    );
}
// ============================================================
// 7. pybind exposed functions
// ============================================================

static bool same_voxels_unordered(
    int a,
    int b,
    const int32_t* vox,
    const int64_t* offsets,
    int dim
) {
    int64_t sa = offsets[a];
    int64_t ea = offsets[a + 1];
    int64_t sb = offsets[b];
    int64_t eb = offsets[b + 1];

    if (ea - sa != eb - sb) return false;

    std::vector<std::vector<int32_t>> va;
    std::vector<std::vector<int32_t>> vb;

    va.reserve(static_cast<size_t>(ea - sa));
    vb.reserve(static_cast<size_t>(eb - sb));

    for (int64_t i = sa; i < ea; ++i) {
        std::vector<int32_t> row(dim);
        for (int d = 0; d < dim; ++d) row[d] = vox[dim * i + d];
        va.push_back(std::move(row));
    }

    for (int64_t i = sb; i < eb; ++i) {
        std::vector<int32_t> row(dim);
        for (int d = 0; d < dim; ++d) row[d] = vox[dim * i + d];
        vb.push_back(std::move(row));
    }

    std::sort(va.begin(), va.end());
    std::sort(vb.begin(), vb.end());

    return va == vb;
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
    if (matches_arr.ndim() != 2 || matches_arr.shape(1) < 2) {
        throw std::runtime_error("matches must have shape (M, >=2).");
    }

    if (vox_flat_arr.ndim() != 2 || vox_flat_arr.shape(1) != dim) {
        throw std::runtime_error("vox_flat shape does not match dim.");
    }

    if (dim != 2 && dim != 3) {
        throw std::runtime_error("Only 2D or 3D voxels are supported.");
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

    if (parents1.shape(0) != n1 || parents2.shape(0) != n2) {
        throw std::runtime_error("frames and parents length mismatch.");
    }

    if (vox_offsets.shape(0) != N + 1) {
        throw std::runtime_error("vox_offsets must have length N + 1.");
    }

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

    std::vector<DetArc> detection_arcs;
    std::vector<TransArc> transition_arcs;

    build_easy_arcs(
        easy_subgraphs,
        edges,
        frames1,
        frames2,
        vox_flat.data(0, 0),
        vox_offsets.data(0),
        dim,
        threshold,
        detection_arcs,
        transition_arcs
    );

    std::vector<std::vector<int>> tracks_easy =
        run_cinda_cpp(detection_arcs, transition_arcs, N);

    return py::make_tuple(
        tracks_easy,
        hard_subgraphs,
        edges
    );
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

    if (dim != 2 && dim != 3) {
        throw std::runtime_error("Only 2D or 3D voxels are supported.");
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

    if (parents1.shape(0) != n1 || parents2.shape(0) != n2) {
        throw std::runtime_error("frames and parents length mismatch.");
    }

    if (vox_offsets.shape(0) != N + 1) {
        throw std::runtime_error("vox_offsets must have length N + 1.");
    }

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

    for (py::ssize_t i = 0; i < det_rows; ++i) {
        const DetArc& a = detection_arcs_vec[static_cast<size_t>(i)];

        det(i, 0) = a.id;
        det(i, 1) = a.c_en;
        det(i, 2) = a.c_ex;
        det(i, 3) = a.c_det;
    }

    auto tr = transition_arcs.mutable_unchecked<2>();

    for (py::ssize_t i = 0; i < trans_rows; ++i) {
        const TransArc& a = transition_arcs_vec[static_cast<size_t>(i)];

        tr(i, 0) = a.src;
        tr(i, 1) = a.dst;
        tr(i, 2) = a.cost;
    }

    return py::make_tuple(
        detection_arcs,
        transition_arcs,
        easy_subgraphs,
        hard_subgraphs,
        edges
    );
}


// ============================================================
// 8. Module binding
// ============================================================

static BaseGraph build_base_graph_cpp(
    const std::vector<int>& hard_ids_input,
    const std::vector<std::array<int, 2>>& matches_hard_input,
    const std::vector<std::vector<int>>& edges,
    const int32_t* vox,
    const int64_t* offsets,
    int dim,
    int N_original,
    double threshold
) {
    constexpr double BIG = 1e3;
    const double threshold2 = -2.0 * threshold - 1e-6;

    BaseGraph base;
    base.N_original = N_original;
    base.threshold = threshold;

    base.node_in.assign(N_original, -1);
    base.node_out.assign(N_original, -1);
    base.node_single.assign(N_original, -1);

    // ------------------------------------------------------------
    // 1. identical merge, same as Python build_base_graph.
    // ------------------------------------------------------------
    std::vector<uint8_t> removed_cell(N_original, 0);
    std::vector<int> kept_of(N_original, -1);
    std::vector<std::array<int, 2>> matches_hard;

    for (size_t mid = 0; mid < matches_hard_input.size(); ++mid) {
        int u = matches_hard_input[mid][0];
        int v = matches_hard_input[mid][1];

        if (same_voxels_unordered(u, v, vox, offsets, dim)) {
            removed_cell[v] = 1;
            kept_of[v] = u;
        } else {
            matches_hard.push_back({u, v});
        }
    }

    base.matches = matches_hard;

    // ------------------------------------------------------------
    // 2. hard_ids after merge.
    // ------------------------------------------------------------
    std::vector<int> hard_ids;
    hard_ids.reserve(hard_ids_input.size());

    std::vector<uint8_t> in_hard(N_original, 0);

    for (int x : hard_ids_input) {
        if (x < 0 || x >= N_original) continue;
        if (removed_cell[x]) continue;
        hard_ids.push_back(x);
        in_hard[x] = 1;
    }

    // ------------------------------------------------------------
    // 3. Build redirected adjacency for this hard subgraph.
    // ------------------------------------------------------------
    std::vector<std::vector<int>> adj(N_original);

    for (int u : hard_ids) {
        if (u < 0 || u >= static_cast<int>(edges.size())) continue;

        for (int v : edges[u]) {
            int vv = v;
            if (vv >= 0 && vv < N_original && removed_cell[vv]) {
                vv = kept_of[vv];
            }

            if (vv < 0 || vv >= N_original) continue;
            if (!in_hard[vv]) continue;
            if (vv == u) continue;

            adj[u].push_back(vv);
        }
    }

    for (int removed = 0; removed < N_original; ++removed) {
        if (!removed_cell[removed]) continue;

        int kept = kept_of[removed];
        if (kept < 0 || kept >= N_original) continue;
        if (!in_hard[kept]) continue;
        if (removed < 0 || removed >= static_cast<int>(edges.size())) continue;

        for (int v : edges[removed]) {
            int vv = v;
            if (vv >= 0 && vv < N_original && removed_cell[vv]) {
                vv = kept_of[vv];
            }
            if (vv < 0 || vv >= N_original) continue;
            if (!in_hard[vv]) continue;
            if (vv == kept) continue;

            adj[kept].push_back(vv);
        }
    }

    for (auto& ns : adj) {
        if (ns.size() > 1) {
            std::sort(ns.begin(), ns.end());
            ns.erase(std::unique(ns.begin(), ns.end()), ns.end());
        }
    }

    // ------------------------------------------------------------
    // 4. matched/unmatched nodes.
    // ------------------------------------------------------------
    std::vector<uint8_t> matched(N_original, 0);

    for (const auto& m : matches_hard) {
        matched[m[0]] = 1;
        matched[m[1]] = 1;
    }

    int next_id = 0;

    auto add_det = [&](double c_en, double c_ex, double c_det) {
        int id = next_id++;
        base.detection.push_back(
            DetArc{
                static_cast<double>(id),
                c_en,
                c_ex,
                c_det
            }
        );
        base.expand_to_original.push_back(-1);
        return id;
    };

    // unmatched single nodes
    for (int u : hard_ids) {
        if (!matched[u]) {
            int nid = add_det(threshold, threshold, threshold2);
            base.node_single[u] = nid;
            base.expand_to_original[nid] = u;
        }
    }

    // matched in/out nodes
    for (const auto& m : matches_hard) {
        for (int cell : {m[0], m[1]}) {
            if (base.node_in[cell] < 0) {
                int in_id = add_det(threshold, BIG, 0.0);
                base.node_in[cell] = in_id;
                base.expand_to_original[in_id] = cell;

                int out_id = add_det(BIG, threshold, 0.0);
                base.node_out[cell] = out_id;
                base.expand_to_original[out_id] = cell;
            }
        }
    }

    // control node per match and four zero-cost arcs
    for (const auto& m : matches_hard) {
        int u = m[0];
        int v = m[1];

        int ctrl = add_det(BIG, BIG, threshold2);

        base.transition.push_back(TransArc{static_cast<double>(base.node_in[u]), static_cast<double>(ctrl), 0.0});
        base.transition.push_back(TransArc{static_cast<double>(base.node_in[v]), static_cast<double>(ctrl), 0.0});
        base.transition.push_back(TransArc{static_cast<double>(ctrl), static_cast<double>(base.node_out[u]), 0.0});
        base.transition.push_back(TransArc{static_cast<double>(ctrl), static_cast<double>(base.node_out[v]), 0.0});
    }

    // inter-cell edges
    for (int u : hard_ids) {
        for (int v : adj[u]) {
            int src = (base.node_out[u] >= 0) ? base.node_out[u] : base.node_single[u];
            int dst = (base.node_in[v] >= 0) ? base.node_in[v] : base.node_single[v];

            if (src < 0 || dst < 0) continue;

            double cost = edge_cost_cpp(u, v, vox, offsets, dim);

            base.transition.push_back(
                TransArc{
                    static_cast<double>(src),
                    static_cast<double>(dst),
                    cost
                }
            );
        }
    }

    return base;
}

static bool flow_node_disabled(
    const BaseGraph& base,
    const BBState& st,
    int flow_node
) {
    if (flow_node < 0 ||
        flow_node >= static_cast<int>(base.expand_to_original.size())) {
        return false;
    }

    int cell = base.expand_to_original[flow_node];

    if (cell < 0) {
        return false;
    }

    return st.forbidden_cell[cell] != 0;
}



static void materialize_graph(
    const BaseGraph& base,
    const BBState& st,
    std::vector<DetArc>& det_out,
    std::vector<TransArc>& trans_out
) {
    constexpr double BIG = 1e3;

    det_out = base.detection;

    for (int cell = 0; cell < static_cast<int>(st.forbidden_cell.size()); ++cell) {
        if (!st.forbidden_cell[cell]) {
            continue;
        }

        int in_id = base.node_in[cell];
        int out_id = base.node_out[cell];

        if (in_id >= 0) {
            det_out[static_cast<size_t>(in_id)].c_en = BIG;
            det_out[static_cast<size_t>(in_id)].c_ex = BIG;
        }

        if (out_id >= 0) {
            det_out[static_cast<size_t>(out_id)].c_en = BIG;
            det_out[static_cast<size_t>(out_id)].c_ex = BIG;
        }
    }

    trans_out.clear();
    trans_out.reserve(base.transition.size());

    for (const auto& a : base.transition) {
        int s = static_cast<int>(a.src);
        int t = static_cast<int>(a.dst);

        if (flow_node_disabled(base, st, s)) continue;
        if (flow_node_disabled(base, st, t)) continue;

        trans_out.push_back(a);
    }
}

static std::vector<int> find_violations_cpp(
    const std::vector<std::vector<int>>& trajectories,
    const BaseGraph& base
) {
    std::vector<uint8_t> appeared(base.N_original, 0);

    for (const auto& path : trajectories) {
        for (int flow_node : path) {
            if (flow_node < 0 || flow_node >= static_cast<int>(base.expand_to_original.size())) {
                continue;
            }

            int cell = base.expand_to_original[flow_node];

            if (cell >= 0 && cell < base.N_original) {
                appeared[cell] = 1;
            }
        }
    }

    std::vector<int> violations;

    for (int mid = 0; mid < static_cast<int>(base.matches.size()); ++mid) {
        int u = base.matches[mid][0];
        int v = base.matches[mid][1];

        if (appeared[u] && appeared[v]) {
            violations.push_back(mid);
        }
    }

    return violations;
}

static FlowResult solve_state_flow(
    const BaseGraph& base,
    const BBState& st
) {
    std::vector<DetArc> det;
    std::vector<TransArc> trans;

    materialize_graph(base, st, det, trans);

    FlowResult res;
    res.trajectories = run_cinda_cpp_with_cost(
        det,
        trans,
        static_cast<int>(base.expand_to_original.size()),
        res.cost
    );

    res.violations = find_violations_cpp(res.trajectories, base);
    res.ok = true;

    return res;
}

static CorrectionData initialize_correction_data_cpp(
    const BaseGraph& base
) {
    CorrectionData cd;

    const int M = static_cast<int>(base.matches.size());
    cd.min_switch_cost.assign(M, 0.0);
    cd.correction.assign(M, 0.0);
    cd.cost_diff.assign(M, 0.0);
    cd.u_switch_cost.assign(M, 0.0);
    cd.v_switch_cost.assign(M, 0.0);

    const int num_flow_nodes = static_cast<int>(base.detection.size());

    std::vector<double> incoming_min(num_flow_nodes, base.threshold);
    std::vector<double> outgoing_min(num_flow_nodes, base.threshold);

    for (const auto& a : base.transition) {
        int s = static_cast<int>(a.src);
        int t = static_cast<int>(a.dst);

        if (s >= 0 && s < num_flow_nodes) {
            outgoing_min[s] = std::min(outgoing_min[s], a.cost);
        }

        if (t >= 0 && t < num_flow_nodes) {
            incoming_min[t] = std::min(incoming_min[t], a.cost);
        }
    }

    for (int mid = 0; mid < M; ++mid) {
        int u = base.matches[mid][0];
        int v = base.matches[mid][1];

        int u_in = base.node_in[u];
        int u_out = base.node_out[u];
        int v_in = base.node_in[v];
        int v_out = base.node_out[v];

        double u_in_min = incoming_min[u_in];
        double u_out_min = outgoing_min[u_out];
        double v_in_min = incoming_min[v_in];
        double v_out_min = outgoing_min[v_out];

        double min_violation_cost = std::max(
            u_in_min + v_out_min,
            v_in_min + u_out_min
        );

        double u_min_cost = u_in_min + u_out_min;
        double v_min_cost = v_in_min + v_out_min;

        double min_feasible_cost = std::min(u_min_cost, v_min_cost);
        double raw_switch = min_feasible_cost - min_violation_cost;

        cd.min_switch_cost[mid] = raw_switch;
        cd.correction[mid] = std::max(raw_switch, 0.0);
        cd.cost_diff[mid] = u_min_cost - v_min_cost;
        cd.u_switch_cost[mid] = std::max(u_min_cost - min_violation_cost, 0.0);
        cd.v_switch_cost[mid] = std::max(v_min_cost - min_violation_cost, 0.0);
    }

    return cd;
}

static double compute_corrected_lower_bound_cpp(
    double current_cost,
    const std::vector<int>& violations,
    const CorrectionData& cd,
    double best_cost
) {
    if (violations.empty()) {
        return current_cost;
    }

    double fix = 0.0;

    for (int mid : violations) {
        if (mid >= 0 && mid < static_cast<int>(cd.correction.size())) {
            fix += cd.correction[mid];
        }
    }

    double lb = current_cost + fix;

    if (lb >= best_cost) {
        return std::numeric_limits<double>::infinity();
    }

    return lb;
}

static int fixed_count(const BBState& st) {
    int c = 0;
    for (int8_t x : st.fixed_choice) {
        if (x >= 0) ++c;
    }
    return c;
}

static std::string state_key(const BBState& st) {
    std::ostringstream oss;

    for (int i = 0; i < static_cast<int>(st.fixed_choice.size()); ++i) {
        if (st.fixed_choice[i] >= 0) {
            oss << i << ":" << static_cast<int>(st.fixed_choice[i]) << ";";
        }
    }

    return oss.str();
}

static std::vector<int> filter_unfixed_violations(
    const std::vector<int>& violations,
    const BBState& st
) {
    std::vector<int> out;
    out.reserve(violations.size());

    for (int mid : violations) {
        if (mid >= 0 &&
            mid < static_cast<int>(st.fixed_choice.size()) &&
            st.fixed_choice[mid] < 0) {
            out.push_back(mid);
        }
    }

    return out;
}

static bool forbid_cell(
    BBState& st,
    const BaseGraph& base,
    int cell
) {
    if (cell < 0 || cell >= base.N_original) {
        return false;
    }

    // matched cells should have both in/out nodes.
    if (base.node_in[cell] < 0 || base.node_out[cell] < 0) {
        return false;
    }

    st.forbidden_cell[cell] = 1;
    return true;
}

static bool forbid_cells(
    BBState& st,
    const BaseGraph& base,
    const std::vector<int>& cells
) {
    for (int c : cells) {
        if (!forbid_cell(st, base, c)) {
            return false;
        }
    }
    return true;
}


static int cell_side(int cell_id, int n1) {
    return cell_id < n1 ? 0 : 1;
}

static int cell_frame(
    int cell_id,
    int n1,
    const py::detail::unchecked_reference<int64_t, 1>& frames1,
    const py::detail::unchecked_reference<int64_t, 1>& frames2
) {
    if (cell_id < n1) {
        return static_cast<int>(frames1(cell_id));
    }
    return static_cast<int>(frames2(cell_id - n1));
}

static long long pair_key_undirected(int a, int b) {
    int x = std::min(a, b);
    int y = std::max(a, b);

    return (static_cast<long long>(x) << 32) ^
           static_cast<unsigned int>(y);
}

static std::vector<ConflictBlock> build_frame_conflict_blocks_cpp(
    const BaseGraph& base,
    int n1,
    const py::detail::unchecked_reference<int64_t, 1>& frames1,
    const py::detail::unchecked_reference<int64_t, 1>& frames2,
    bool star_only,
    std::vector<int>& match_to_block
) {
    const int M = static_cast<int>(base.matches.size());

    match_to_block.assign(M, -1);

    std::unordered_map<int, std::unordered_map<int, std::vector<int>>> frame_adj;
    std::unordered_map<long long, int> pair_to_match;

    for (int mid = 0; mid < M; ++mid) {
        int u = base.matches[static_cast<size_t>(mid)][0];
        int v = base.matches[static_cast<size_t>(mid)][1];

        pair_to_match[pair_key_undirected(u, v)] = mid;

        int fu = cell_frame(u, n1, frames1, frames2);
        int fv = cell_frame(v, n1, frames1, frames2);

        if (fu != fv) {
            continue;
        }

        if (cell_side(u, n1) == cell_side(v, n1)) {
            continue;
        }

        frame_adj[fu][u].push_back(v);
        frame_adj[fu][v].push_back(u);
    }

    std::vector<int> frames;
    frames.reserve(frame_adj.size());

    for (const auto& kv : frame_adj) {
        frames.push_back(kv.first);
    }

    std::sort(frames.begin(), frames.end());

    std::vector<ConflictBlock> blocks;

    for (int f : frames) {
        auto& adj = frame_adj[f];
        std::unordered_set<int> visited;

        for (const auto& kv : adj) {
            int start = kv.first;

            if (visited.find(start) != visited.end()) {
                continue;
            }

            std::vector<int> q;
            std::vector<int> comp;

            q.push_back(start);
            visited.insert(start);

            for (size_t qi = 0; qi < q.size(); ++qi) {
                int x = q[qi];
                comp.push_back(x);

                for (int y : adj[x]) {
                    if (visited.insert(y).second) {
                        q.push_back(y);
                    }
                }
            }

            if (comp.size() < 2) {
                continue;
            }

            std::vector<int> left_nodes;
            std::vector<int> right_nodes;

            for (int x : comp) {
                if (cell_side(x, n1) == 0) {
                    left_nodes.push_back(x);
                } else {
                    right_nodes.push_back(x);
                }
            }

            std::sort(left_nodes.begin(), left_nodes.end());
            std::sort(right_nodes.begin(), right_nodes.end());

            if (left_nodes.empty() || right_nodes.empty()) {
                continue;
            }

            std::vector<int> mids;

            for (int u : left_nodes) {
                for (int v : right_nodes) {
                    auto it = pair_to_match.find(pair_key_undirected(u, v));
                    if (it != pair_to_match.end()) {
                        mids.push_back(it->second);
                    }
                }
            }

            std::sort(mids.begin(), mids.end());
            mids.erase(std::unique(mids.begin(), mids.end()), mids.end());

            if (mids.size() <= 1) {
                continue;
            }

            if (star_only && !(left_nodes.size() == 1 || right_nodes.size() == 1)) {
                continue;
            }

            ConflictBlock block;
            block.block_id = static_cast<int>(blocks.size());
            block.frame = f;
            block.left_nodes = left_nodes;
            block.right_nodes = right_nodes;
            block.match_ids = mids;

            if (left_nodes.size() == 1 && right_nodes.size() > 1) {
                block.kind = 0;  // 1-vs-k
            } else if (left_nodes.size() > 1 && right_nodes.size() == 1) {
                block.kind = 1;  // k-vs-1
            } else {
                block.kind = 2;  // k-vs-m
            }

            for (int mid : mids) {
                match_to_block[static_cast<size_t>(mid)] = block.block_id;
            }

            blocks.push_back(std::move(block));
        }
    }

    return blocks;
}

static const ConflictBlock* select_conflict_block_cpp(
    const std::vector<int>& violations,
    const BBState& st,
    const std::vector<ConflictBlock>& blocks,
    const std::vector<int>& match_to_block,
    const CorrectionData& cd
) {
    if (violations.empty() || blocks.empty()) {
        return nullptr;
    }

    std::unordered_set<int> unresolved;
    unresolved.reserve(violations.size());

    for (int mid : violations) {
        if (mid >= 0 &&
            mid < static_cast<int>(st.fixed_choice.size()) &&
            st.fixed_choice[static_cast<size_t>(mid)] < 0) {
            unresolved.insert(mid);
        }
    }

    if (unresolved.empty()) {
        return nullptr;
    }

    std::unordered_set<int> candidate_block_ids;

    for (int mid : unresolved) {
        if (mid >= 0 && mid < static_cast<int>(match_to_block.size())) {
            int bid = match_to_block[static_cast<size_t>(mid)];
            if (bid >= 0) {
                candidate_block_ids.insert(bid);
            }
        }
    }

    const ConflictBlock* best_block = nullptr;

    int best_cover = -1;
    double best_impact = -1.0;
    int best_num_matches = -1;
    int best_neg_frame = std::numeric_limits<int>::min();

    for (int bid : candidate_block_ids) {
        if (bid < 0 || bid >= static_cast<int>(blocks.size())) {
            continue;
        }

        const ConflictBlock& block = blocks[static_cast<size_t>(bid)];

        int cover = 0;
        double impact = 0.0;

        for (int mid : block.match_ids) {
            if (unresolved.find(mid) != unresolved.end()) {
                ++cover;

                if (mid >= 0 &&
                    mid < static_cast<int>(cd.min_switch_cost.size())) {
                    impact += std::max(cd.min_switch_cost[static_cast<size_t>(mid)], 0.0);
                }
            }
        }

        if (cover <= 0) {
            continue;
        }

        int num_matches = static_cast<int>(block.match_ids.size());
        int neg_frame = -block.frame;

        bool better = false;

        if (cover > best_cover) better = true;
        else if (cover == best_cover && impact > best_impact) better = true;
        else if (cover == best_cover && impact == best_impact && num_matches > best_num_matches) better = true;
        else if (cover == best_cover && impact == best_impact &&
                 num_matches == best_num_matches && neg_frame > best_neg_frame) better = true;

        if (better) {
            best_cover = cover;
            best_impact = impact;
            best_num_matches = num_matches;
            best_neg_frame = neg_frame;
            best_block = &block;
        }
    }

    return best_block;
}
static int assignment_for_forbid_node(
    int mid,
    int forbid_node,
    const BaseGraph& base
) {
    int u = base.matches[static_cast<size_t>(mid)][0];
    int v = base.matches[static_cast<size_t>(mid)][1];

    // choice=0 means forbid v.
    if (forbid_node == v) return 0;

    // choice=1 means forbid u.
    if (forbid_node == u) return 1;

    return -1;
}

static bool build_block_child_cpp(
    const BBState& parent,
    const BaseGraph& base,
    const ConflictBlock& block,
    int choose_side,
    BBState& child
) {
    child = parent;
    child.depth = parent.depth + 1;

    std::vector<int> forbid_nodes;

    if (choose_side == 0) {
        // choose left/M1 side, forbid right/M2 side
        forbid_nodes = block.right_nodes;
    } else if (choose_side == 1) {
        // choose right/M2 side, forbid left/M1 side
        forbid_nodes = block.left_nodes;
    } else {
        return false;
    }

    std::unordered_set<int> forbid_set(
        forbid_nodes.begin(),
        forbid_nodes.end()
    );

    for (int mid : block.match_ids) {
        if (mid < 0 || mid >= static_cast<int>(base.matches.size())) {
            return false;
        }

        int u = base.matches[static_cast<size_t>(mid)][0];
        int v = base.matches[static_cast<size_t>(mid)][1];

        bool fu = forbid_set.find(u) != forbid_set.end();
        bool fv = forbid_set.find(v) != forbid_set.end();

        if (fu && fv) return false;
        if (!fu && !fv) return false;

        int forbid_node = fu ? u : v;
        int choice = assignment_for_forbid_node(mid, forbid_node, base);

        if (choice < 0) return false;

        if (child.fixed_choice[static_cast<size_t>(mid)] >= 0 &&
            child.fixed_choice[static_cast<size_t>(mid)] != choice) {
            return false;
        }

        child.fixed_choice[static_cast<size_t>(mid)] =
            static_cast<int8_t>(choice);
    }

    return forbid_cells(child, base, forbid_nodes);
}

static std::vector<int> select_batch_violations_cpp(
    const std::vector<int>& violations,
    const BBState& st,
    const CorrectionData& cd,
    int batch_size
) {
    std::vector<int> mids;

    for (int mid : violations) {
        if (mid >= 0 &&
            mid < static_cast<int>(st.fixed_choice.size()) &&
            st.fixed_choice[static_cast<size_t>(mid)] < 0) {
            mids.push_back(mid);
        }
    }

    if (mids.empty()) {
        return mids;
    }

    batch_size = std::max(1, std::min(batch_size, static_cast<int>(mids.size())));

    std::sort(
        mids.begin(),
        mids.end(),
        [&](int a, int b) {
            double sa = 0.0;
            double sb = 0.0;

            if (a >= 0 && a < static_cast<int>(cd.min_switch_cost.size())) {
                sa = cd.min_switch_cost[static_cast<size_t>(a)];
            }

            if (b >= 0 && b < static_cast<int>(cd.min_switch_cost.size())) {
                sb = cd.min_switch_cost[static_cast<size_t>(b)];
            }

            return sa > sb;
        }
    );

    mids.resize(static_cast<size_t>(batch_size));
    return mids;
}

static bool build_batch_child_cpp(
    const BBState& parent,
    const BaseGraph& base,
    const std::vector<int>& selected_matches,
    int bitmask,
    BBState& child
) {
    child = parent;
    child.depth = parent.depth + 1;

    std::vector<int> forbid_nodes;

    for (int i = 0; i < static_cast<int>(selected_matches.size()); ++i) {
        int mid = selected_matches[static_cast<size_t>(i)];

        if (mid < 0 || mid >= static_cast<int>(base.matches.size())) {
            return false;
        }

        int choice = (bitmask >> i) & 1;

        if (child.fixed_choice[static_cast<size_t>(mid)] >= 0 &&
            child.fixed_choice[static_cast<size_t>(mid)] != choice) {
            return false;
        }

        int u = base.matches[static_cast<size_t>(mid)][0];
        int v = base.matches[static_cast<size_t>(mid)][1];

        // choice=0: forbid v; choice=1: forbid u
        int forbid_node = (choice == 0) ? v : u;

        child.fixed_choice[static_cast<size_t>(mid)] =
            static_cast<int8_t>(choice);

        forbid_nodes.push_back(forbid_node);
    }

    return forbid_cells(child, base, forbid_nodes);
}

static double compute_corrected_lower_bound_by_blocks_cpp(
    double current_cost,
    const std::vector<int>& violations,
    const CorrectionData& cd,
    double best_cost,
    const std::vector<ConflictBlock>& blocks,
    const std::vector<int>& match_to_block
) {
    if (violations.empty()) {
        return current_cost;
    }

    if (blocks.empty() || match_to_block.empty()) {
        return compute_corrected_lower_bound_cpp(
            current_cost,
            violations,
            cd,
            best_cost
        );
    }

    std::unordered_set<int> viol_set(
        violations.begin(),
        violations.end()
    );

    std::vector<uint8_t> used_block(blocks.size(), 0);
    double min_fix_cost = 0.0;

    for (int mid : viol_set) {
        int bid = -1;

        if (mid >= 0 && mid < static_cast<int>(match_to_block.size())) {
            bid = match_to_block[static_cast<size_t>(mid)];
        }

        if (bid < 0) {
            if (mid >= 0 && mid < static_cast<int>(cd.correction.size())) {
                min_fix_cost += cd.correction[static_cast<size_t>(mid)];
            }
            continue;
        }

        if (bid >= static_cast<int>(blocks.size())) {
            continue;
        }

        if (used_block[static_cast<size_t>(bid)]) {
            continue;
        }

        used_block[static_cast<size_t>(bid)] = 1;

        const ConflictBlock& block = blocks[static_cast<size_t>(bid)];

        double max_val = 0.0;
        bool has_val = false;

        for (int m : block.match_ids) {
            if (viol_set.find(m) != viol_set.end() &&
                m >= 0 &&
                m < static_cast<int>(cd.correction.size())) {
                max_val = std::max(max_val, cd.correction[static_cast<size_t>(m)]);
                has_val = true;
            }
        }

        if (has_val) {
            min_fix_cost += max_val;
        }
    }

    double lb = current_cost + min_fix_cost;

    if (lb >= best_cost) {
        return std::numeric_limits<double>::infinity();
    }

    return lb;
}

struct GreedyResult {
    bool found = false;
    double cost = std::numeric_limits<double>::infinity();
    std::vector<std::vector<int>> solution;
    int flow_calls = 0;
};

static GreedyResult find_feasible_solution_greedy_cpp(
    const BaseGraph& base,
    const FlowResult& root_flow,
    const BBState& root_state,
    int max_iter
) {
    GreedyResult gr;

    if (!root_flow.ok) {
        return gr;
    }

    BBState current_state = root_state;
    FlowResult current_flow = root_flow;

    for (int it = 0; it < max_iter; ++it) {
        std::vector<int> cur_viol =
            filter_unfixed_violations(current_flow.violations, current_state);

        if (cur_viol.empty()) {
            gr.found = true;
            gr.cost = current_flow.cost;
            gr.solution = current_flow.trajectories;
            return gr;
        }

        int mid = cur_viol.front();

        struct Candidate {
            int remain;
            double cost;
            BBState state;
            FlowResult flow;
        };

        std::vector<Candidate> candidates;

        for (int choice : {0, 1}) {
            BBState child;

            std::vector<int> selected = {mid};

            if (!build_batch_child_cpp(
                    current_state,
                    base,
                    selected,
                    choice,
                    child)) {
                continue;
            }

            FlowResult child_flow = solve_state_flow(base, child);
            gr.flow_calls += 1;

            if (!child_flow.ok) {
                continue;
            }

            std::vector<int> child_viol =
                filter_unfixed_violations(child_flow.violations, child);

            candidates.push_back(
                Candidate{
                    static_cast<int>(child_viol.size()),
                    child_flow.cost,
                    std::move(child),
                    std::move(child_flow)
                }
            );
        }

        if (candidates.empty()) {
            return gr;
        }

        std::sort(
            candidates.begin(),
            candidates.end(),
            [](const Candidate& a, const Candidate& b) {
                if (a.remain != b.remain) return a.remain < b.remain;
                return a.cost < b.cost;
            }
        );

        current_state = std::move(candidates[0].state);
        current_flow = std::move(candidates[0].flow);

        if (candidates[0].remain == 0) {
            gr.found = true;
            gr.cost = current_flow.cost;
            gr.solution = current_flow.trajectories;
            return gr;
        }
    }

    return gr;
}

static HardResult solve_hard_subgraph_cpp_optimized(
    const std::vector<int>& hard_ids,
    const std::vector<std::array<int, 2>>& matches_hard,
    const std::vector<std::vector<int>>& edges,
    const int32_t* vox,
    const int64_t* offsets,
    int dim,
    int N_original,
    double threshold,
    int hard_id,
    int max_bb_nodes,
    int batch_branch_size,
    bool use_initial_greedy,
    bool use_conflict_blocks,
    bool conflict_blocks_star_only,
    int n1,
    const py::detail::unchecked_reference<int64_t, 1>& frames1,
    const py::detail::unchecked_reference<int64_t, 1>& frames2,
    double tol
) {
    auto t0_all = std::chrono::steady_clock::now();

    BaseGraph base = build_base_graph_cpp(
        hard_ids,
        matches_hard,
        edges,
        vox,
        offsets,
        dim,
        N_original,
        threshold
    );

    BBState root;
    root.fixed_choice.assign(base.matches.size(), -1);
    root.forbidden_cell.assign(N_original, 0);
    root.depth = 0;

    int flow_calls = 0;
    int explored = 0;
    int max_depth = 0;
    int block_branches = 0;
    int fallback_branches = 0;

    FlowResult root_flow = solve_state_flow(base, root);
    flow_calls += 1;

    if (!root_flow.ok) {
        py::dict stats;
        stats["hard_id"] = hard_id;
        stats["num_nodes"] = static_cast<int>(hard_ids.size());
        stats["num_matches"] = static_cast<int>(base.matches.size());
        stats["root_satisfied_matches"] = 0;
        stats["root_violated_matches"] = static_cast<int>(base.matches.size());
        stats["max_depth"] = 0;
        stats["branches_explored"] = 0;
        stats["flow_calls"] = flow_calls;
        stats["block_branches"] = 0;
        stats["fallback_branches"] = 0;
        stats["certified_optimal"] = false;
        stats["termination_reason"] = "root_flow_failed";
        stats["time_solve_flow"] = 0.0;

        return HardResult{{}, stats};
    }

    std::vector<int> root_violations =
        filter_unfixed_violations(root_flow.violations, root);

    int num_matches = static_cast<int>(base.matches.size());
    int root_violated_matches = static_cast<int>(root_violations.size());
    int root_satisfied_matches = num_matches - root_violated_matches;

    CorrectionData cd = initialize_correction_data_cpp(base);

    std::vector<int> match_to_block;
    std::vector<ConflictBlock> conflict_blocks;

    if (use_conflict_blocks) {
        conflict_blocks = build_frame_conflict_blocks_cpp(
            base,
            n1,
            frames1,
            frames2,
            conflict_blocks_star_only,
            match_to_block
        );
    } else {
        match_to_block.assign(num_matches, -1);
    }

    double best_cost = std::numeric_limits<double>::infinity();
    std::vector<std::vector<int>> best_solution;

    if (root_violations.empty()) {
        best_cost = root_flow.cost;
        best_solution = root_flow.trajectories;
    } else if (use_initial_greedy) {
        GreedyResult gr = find_feasible_solution_greedy_cpp(
            base,
            root_flow,
            root,
            100
        );

        flow_calls += gr.flow_calls;

        if (gr.found && gr.cost < best_cost) {
            best_cost = gr.cost;
            best_solution = std::move(gr.solution);
        }
    }

    double corrected_root_lb = compute_corrected_lower_bound_by_blocks_cpp(
        root_flow.cost,
        root_violations,
        cd,
        best_cost,
        conflict_blocks,
        match_to_block
    );

    std::vector<BBState> states;
    states.reserve(2048);
    states.push_back(root);

    std::priority_queue<PQItem> pq;
    int counter = 0;

    std::unordered_set<std::string> seen_states;
    seen_states.reserve(4096);

    std::unordered_map<std::string, FlowResult> flow_cache;
    flow_cache.reserve(4096);

    std::string root_key = state_key(root);
    seen_states.insert(root_key);
    flow_cache[root_key] = root_flow;

    if (std::isfinite(corrected_root_lb) &&
        corrected_root_lb < best_cost - tol) {
        pq.push(PQItem{corrected_root_lb, counter++, 0});
    }

    bool certified_optimal = false;
    std::string termination_reason = "unknown";
    double final_min_lb = corrected_root_lb;

    while (!pq.empty() && explored < max_bb_nodes) {
        PQItem item = pq.top();
        pq.pop();

        double lb = item.lb;
        final_min_lb = lb;

        if (lb >= best_cost - tol) {
            certified_optimal = true;
            termination_reason = "min_heap_lb_reached_best";
            break;
        }

        BBState node = states[static_cast<size_t>(item.state_id)];
        explored += 1;
        max_depth = std::max(max_depth, node.depth);

        std::string key = state_key(node);

        FlowResult flow;

        auto it_cache = flow_cache.find(key);
        if (it_cache != flow_cache.end()) {
            flow = it_cache->second;
        } else {
            flow = solve_state_flow(base, node);
            flow_calls += 1;
            flow_cache[key] = flow;
        }

        if (!flow.ok) {
            continue;
        }

        std::vector<int> violations =
            filter_unfixed_violations(flow.violations, node);

        if (violations.empty()) {
            if (flow.cost < best_cost - tol) {
                best_cost = flow.cost;
                best_solution = flow.trajectories;
            }
            continue;
        }

        if (flow.cost >= best_cost - tol) {
            continue;
        }

        std::vector<BBState> children;

        // ============================================================
        // 1. Prefer conflict-block branching.
        // ============================================================
        const ConflictBlock* block = nullptr;

        if (use_conflict_blocks && !conflict_blocks.empty()) {
            block = select_conflict_block_cpp(
                violations,
                node,
                conflict_blocks,
                match_to_block,
                cd
            );
        }

        if (block != nullptr) {
            block_branches += 1;

            for (int choose_side : {0, 1}) {
                BBState child;

                if (!build_block_child_cpp(
                        node,
                        base,
                        *block,
                        choose_side,
                        child)) {
                    continue;
                }

                std::string child_key = state_key(child);

                if (seen_states.find(child_key) != seen_states.end()) {
                    continue;
                }

                seen_states.insert(child_key);
                children.push_back(std::move(child));
            }
        }

        // ============================================================
        // 2. Fallback batch branching.
        // ============================================================
        if (children.empty()) {
            fallback_branches += 1;

            std::vector<int> selected =
                select_batch_violations_cpp(
                    violations,
                    node,
                    cd,
                    batch_branch_size
                );

            if (selected.empty()) {
                continue;
            }

            int num_assignments = 1 << static_cast<int>(selected.size());

            for (int mask = 0; mask < num_assignments; ++mask) {
                BBState child;

                if (!build_batch_child_cpp(
                        node,
                        base,
                        selected,
                        mask,
                        child)) {
                    continue;
                }

                std::string child_key = state_key(child);

                if (seen_states.find(child_key) != seen_states.end()) {
                    continue;
                }

                seen_states.insert(child_key);
                children.push_back(std::move(child));
            }
        }

        // ============================================================
        // 3. Solve children and push promising states.
        // ============================================================
        for (BBState& child : children) {
            std::string child_key = state_key(child);

            FlowResult child_flow = solve_state_flow(base, child);
            flow_calls += 1;

            flow_cache[child_key] = child_flow;

            if (!child_flow.ok) {
                continue;
            }

            std::vector<int> child_violations =
                filter_unfixed_violations(child_flow.violations, child);

            if (child_violations.empty()) {
                if (child_flow.cost < best_cost - tol) {
                    best_cost = child_flow.cost;
                    best_solution = child_flow.trajectories;
                }
                continue;
            }

            double child_lb = compute_corrected_lower_bound_by_blocks_cpp(
                child_flow.cost,
                child_violations,
                cd,
                best_cost,
                conflict_blocks,
                match_to_block
            );

            if (!std::isfinite(child_lb) ||
                child_lb >= best_cost - tol) {
                continue;
            }

            int child_id = static_cast<int>(states.size());
            states.push_back(std::move(child));
            pq.push(PQItem{child_lb, counter++, child_id});
        }
    }

    if (certified_optimal) {
        // already set
    } else if (pq.empty()) {
        certified_optimal = true;
        termination_reason = "priority_queue_empty";
        final_min_lb = best_cost;
    } else if (explored >= max_bb_nodes) {
        certified_optimal = false;
        termination_reason = "max_bb_nodes_reached";
        final_min_lb = pq.top().lb;
    } else {
        certified_optimal = false;
        termination_reason = "unknown_exit";
        final_min_lb = pq.empty() ? best_cost : pq.top().lb;
    }

    std::vector<std::vector<int>> recovered;

    if (std::isfinite(best_cost)) {
        for (const auto& path : best_solution) {
            std::vector<int> cells;

            for (int flow_node : path) {
                if (flow_node < 0 ||
                    flow_node >= static_cast<int>(base.expand_to_original.size())) {
                    continue;
                }

                int cell = base.expand_to_original[static_cast<size_t>(flow_node)];

                if (cell >= 0) {
                    bool exists = false;
                    for (int old : cells) {
                        if (old == cell) {
                            exists = true;
                            break;
                        }
                    }

                    if (!exists) {
                        cells.push_back(cell);
                    }
                }
            }

            if (!cells.empty()) {
                recovered.push_back(std::move(cells));
            }
        }
    }

    auto t1_all = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1_all - t0_all).count();

    double gap = 0.0;
    if (std::isfinite(best_cost) && std::isfinite(final_min_lb)) {
        gap = std::max(0.0, best_cost - final_min_lb);
    }

    py::dict stats;
    stats["hard_id"] = hard_id;
    stats["num_nodes"] = static_cast<int>(hard_ids.size());
    stats["num_matches"] = num_matches;

    stats["root_satisfied_matches"] = root_satisfied_matches;
    stats["root_violated_matches"] = root_violated_matches;

    stats["max_depth"] = max_depth;
    stats["branches_explored"] = explored;
    stats["flow_calls"] = flow_calls;

    stats["block_branches"] = block_branches;
    stats["fallback_branches"] = fallback_branches;

    stats["certified_optimal"] = certified_optimal;
    stats["termination_reason"] = termination_reason;
    stats["time_solve_flow"] = elapsed;

    stats["best_cost"] = best_cost;
    stats["final_min_lb"] = final_min_lb;
    stats["gap"] = gap;
    stats["num_conflict_blocks"] = static_cast<int>(conflict_blocks.size());
    stats["seen_states"] = static_cast<int>(seen_states.size());
    stats["pq_size"] = static_cast<int>(pq.size());

    HardResult out;
    out.tracks = std::move(recovered);
    out.stats = stats;
    return out;
}

py::tuple solve_fusion_cpp(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> frames1_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> parents1_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> frames2_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> parents2_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> matches_arr,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> vox_flat_arr,
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> vox_offsets_arr,
    int dim,
    double threshold,
    int max_bb_nodes,
    int batch_branch_size,
    bool use_initial_greedy,
    bool use_conflict_blocks,
    bool conflict_blocks_star_only
) {
    (void)batch_branch_size;
    (void)use_initial_greedy;
    (void)use_conflict_blocks;
    (void)conflict_blocks_star_only;

    if (matches_arr.ndim() != 2 || matches_arr.shape(1) < 2) {
        throw std::runtime_error("matches must have shape (M, >=2).");
    }

    if (vox_flat_arr.ndim() != 2 || vox_flat_arr.shape(1) != dim) {
        throw std::runtime_error("vox_flat shape does not match dim.");
    }

    if (dim != 2 && dim != 3) {
        throw std::runtime_error("Only 2D or 3D voxels are supported.");
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

    if (parents1.shape(0) != n1 || parents2.shape(0) != n2) {
        throw std::runtime_error("frames and parents length mismatch.");
    }

    if (vox_offsets.shape(0) != N + 1) {
        throw std::runtime_error("vox_offsets must have length N + 1.");
    }

    auto t0 = std::chrono::steady_clock::now();

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

    // 4-2 easy
    std::vector<DetArc> easy_detection;
    std::vector<TransArc> easy_transition;

    build_easy_arcs(
        easy_subgraphs,
        edges,
        frames1,
        frames2,
        vox_flat.data(0, 0),
        vox_offsets.data(0),
        dim,
        threshold,
        easy_detection,
        easy_transition
    );

    std::vector<std::vector<int>> tracks_easy =
        run_cinda_cpp(easy_detection, easy_transition, N);

    // matches: original second column is local movie2 id, convert to global id.
    std::vector<std::array<int, 2>> matches_global;
    matches_global.reserve(static_cast<size_t>(matches.shape(0)));

    for (int i = 0; i < static_cast<int>(matches.shape(0)); ++i) {
        int u = static_cast<int>(matches(i, 0));
        int v = static_cast<int>(matches(i, 1)) + n1;
        matches_global.push_back({u, v});
    }

    std::vector<std::vector<int>> tracks_all = tracks_easy;
    std::vector<py::dict> hard_stats;

    for (int hid = 0; hid < static_cast<int>(hard_subgraphs.size()); ++hid) {
        const auto& nodes = hard_subgraphs[hid];

        std::vector<uint8_t> in_current(N, 0);
        for (int x : nodes) {
            if (x >= 0 && x < N) {
                in_current[x] = 1;
            }
        }

        std::vector<std::array<int, 2>> matches_hard;
        for (const auto& m : matches_global) {
            if (in_current[m[0]] || in_current[m[1]]) {
                matches_hard.push_back(m);
            }
        }

        HardResult hr = solve_hard_subgraph_cpp_optimized(
            nodes,
            matches_hard,
            edges,
            vox_flat.data(0, 0),
            vox_offsets.data(0),
            dim,
            N,
            threshold,
            hid,
            max_bb_nodes,
            batch_branch_size,
            use_initial_greedy,
            use_conflict_blocks,
            conflict_blocks_star_only,
            n1,
            frames1,
            frames2,
            1e-6
        );

        for (auto& tr : hr.tracks) {
            tracks_all.push_back(std::move(tr));
        }

        hard_stats.push_back(hr.stats);
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    int easy_total_nodes = 0;
    std::vector<int> hard_node_sizes;

    for (const auto& g : easy_subgraphs) {
        easy_total_nodes += static_cast<int>(g.size());
    }

    for (const auto& g : hard_subgraphs) {
        hard_node_sizes.push_back(static_cast<int>(g.size()));
    }

    py::dict summary;
    summary["total_step4_5_time_sec"] = elapsed;
    summary["num_easy_subgraphs"] = static_cast<int>(easy_subgraphs.size());
    summary["num_hard_subgraphs"] = static_cast<int>(hard_subgraphs.size());
    summary["easy_total_nodes"] = easy_total_nodes;
    summary["hard_node_sizes"] = hard_node_sizes;

    return py::make_tuple(tracks_all, summary, hard_stats);
}


PYBIND11_MODULE(fast_full_fusion, m) {
    m.doc() = "Fast C++ implementation of INTACT Step 4-1 and Step 4-2 easy-subgraph fusion.";

    m.def(
        "solve_easy_fusion_cpp",
        &solve_easy_fusion_cpp,
        py::arg("frames1"),
        py::arg("parents1"),
        py::arg("frames2"),
        py::arg("parents2"),
        py::arg("matches"),
        py::arg("vox_flat"),
        py::arg("vox_offsets"),
        py::arg("dim"),
        py::arg("threshold")
    );

    m.def(
        "build_easy_arcs_debug_cpp",
        &build_easy_arcs_debug_cpp,
        py::arg("frames1"),
        py::arg("parents1"),
        py::arg("frames2"),
        py::arg("parents2"),
        py::arg("matches"),
        py::arg("vox_flat"),
        py::arg("vox_offsets"),
        py::arg("dim"),
        py::arg("threshold")
    );
    m.def(
        "solve_fusion_cpp",
        &solve_fusion_cpp,
        py::arg("frames1"),
        py::arg("parents1"),
        py::arg("frames2"),
        py::arg("parents2"),
        py::arg("matches"),
        py::arg("vox_flat"),
        py::arg("vox_offsets"),
        py::arg("dim"),
        py::arg("threshold"),
        py::arg("max_bb_nodes") = 1000000,
        py::arg("batch_branch_size") = 2,
        py::arg("use_initial_greedy") = true,
        py::arg("use_conflict_blocks") = true,
        py::arg("conflict_blocks_star_only") = true
    );
}