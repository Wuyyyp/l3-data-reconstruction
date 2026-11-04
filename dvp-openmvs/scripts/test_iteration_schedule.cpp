#include "../IterationSchedule.h"
#include <cassert>
#include <iostream>

int main() {
    const auto old = MakeIterationSchedule(2, "legacy");
    const auto sync = MakeIterationSchedule(2, "legacy-sync");
    const auto balanced = MakeIterationSchedule(2, "balanced");
    assert(old.size() == 8 && sync.size() == 8 && balanced.size() == 7);
    int legacy_steps = 0, balanced_steps = 0;
    for (size_t i = 0; i < old.size(); ++i) {
        assert(old[i].inner_iterations == 3);
        assert(old[i].scale_index == sync[i].scale_index);
        assert(old[i].geometry_index == sync[i].geometry_index);
        assert(old[i].inner_iterations == sync[i].inner_iterations);
        legacy_steps += old[i].inner_iterations;
    }
    const int expected[] = {3, 3, 3, 3, 3, 3, 2};
    for (size_t i = 0; i < balanced.size(); ++i) {
        assert(balanced[i].inner_iterations == expected[i]);
        balanced_steps += balanced[i].inner_iterations;
    }
    assert(legacy_steps == 24 && balanced_steps == 20);
    for (int levels = 1; levels <= 4; ++levels) {
        const auto plan = MakeIterationSchedule(levels, "balanced");
        for (int s = 0; s < levels; ++s) {
            assert(plan[s * 3].scale_index == s);
            assert(plan[s * 3].geometry_index == -1);
            assert(plan[s * 3].inner_iterations == 3);
            assert(plan[s * 3 + 1].weak_peak_radius == 4);
            assert(plan[s * 3 + 2].weak_peak_radius == 2);
        }
        assert(plan.size() == static_cast<size_t>(3 * levels + 1));
        assert(plan[plan.size() - 2].inner_iterations == 3);
        assert(plan.back().geometry_index == 2 && plan.back().inner_iterations == 2);
    }
    const auto short_plan = MakeIterationSchedule(2, "legacy19");
    const int short_inner[] = {3, 3, 3, 3, 3, 2, 2};
    assert(short_plan.size() == 7);
    int short_steps = 0;
    for (int i = 0; i < 7; ++i) {
        assert(short_plan[i].scale_index == i / 4);
        assert(short_plan[i].geometry_index == i % 4 - 1);
        assert(short_plan[i].inner_iterations == short_inner[i]);
        assert(short_plan[i].weak_peak_radius == (i % 4 == 0 ? 6 : i % 4 == 1 ? 4 : 2));
        short_steps += short_plan[i].inner_iterations;
    }
    assert(short_steps == 19);
    assert(UsesInPlaceDepths("legacy") && UsesInPlaceDepths("legacy19"));
    assert(!UsesInPlaceDepths("legacy-sync") && !UsesInPlaceDepths("openmvs"));
    bool invalid_short_scales = false;
    try { MakeIterationSchedule(3, "legacy19"); }
    catch (const std::invalid_argument&) { invalid_short_scales = true; }
    assert(invalid_short_scales);
    const auto aligned = MakeIterationSchedule(3, "openmvs");
    assert(aligned.size() == 5);
    int aligned_steps = 0;
    for (int i = 0; i < 5; ++i) {
        assert(aligned[i].scale_index == (i < 3 ? i : 2));
        assert(aligned[i].geometry_index == (i < 3 ? -1 : i - 3));
        assert(aligned[i].inner_iterations == (i < 3 ? 4 : 1));
        aligned_steps += aligned[i].inner_iterations;
    }
    assert(aligned_steps == 14);
    bool bad_scales = false;
    try { MakeIterationSchedule(2, "openmvs"); }
    catch (const std::invalid_argument&) { bad_scales = true; }
    assert(bad_scales);
    bool rejected = false;
    try { MakeIterationSchedule(2, "typo"); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    std::cout << "PASS: legacy schedule, prior transitions, final geometry and invalid profile\n";
}
