#ifndef DVP_ITERATION_SCHEDULE_H
#define DVP_ITERATION_SCHEDULE_H

#include <stdexcept>
#include <string>
#include <vector>

struct IterationPass {
    int scale_index;
    int geometry_index; // -1 means initialization / upsampling refinement.
    int inner_iterations;
    int weak_peak_radius;
};

inline std::vector<IterationPass> MakeIterationSchedule(int scales, const std::string& profile) {
    if (scales < 1) throw std::invalid_argument("At least one DVP scale is required");
    if (profile != "legacy" && profile != "legacy-sync" && profile != "balanced" && profile != "openmvs" && profile != "legacy19")
        throw std::invalid_argument("DVP_ITERATION_PROFILE must be legacy, legacy-sync, balanced, openmvs or legacy19");
    if (profile == "legacy19" && scales != 2)
        throw std::invalid_argument("legacy19 requires exactly two scales");
    std::vector<IterationPass> passes;
    if (profile == "openmvs") {
        if (scales != 3) throw std::invalid_argument("OpenMVS profile requires three scales");
        for (int scale = 0; scale < scales; ++scale)
            passes.push_back({scale, -1, 4, 6});
        passes.push_back({scales - 1, 0, 1, 4});
        passes.push_back({scales - 1, 1, 1, 2});
        return passes;
    }
    for (int scale = 0; scale < scales; ++scale) {
        // Preserve the full MoGe initialization and every coarse-to-fine transition.
        passes.push_back({scale, -1, 3, 6});
        const int rounds = (profile == "legacy19" && scale == scales - 1) || (profile == "balanced" && scale < scales - 1) ? 2 : 3;
        for (int geo = 0; geo < rounds; ++geo) {
            // legacy19 keeps two final-scale geometric passes with two updates each.
            // Other profiles retain their existing geometry and cleanup policy.
            const int inner = (profile == "legacy19" && scale == scales - 1) || (profile == "balanced" && scale == scales - 1 && geo == 2) ? 2 : 3;
            passes.push_back({scale, geo, inner, geo == 0 ? 4 : 2});
        }
    }
    return passes;
}

inline bool UsesInPlaceDepths(const std::string& profile) {
    return profile == "legacy" || profile == "legacy19";
}

#endif
