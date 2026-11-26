import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT, SINGLES, DURATIONS, OUT = (Path(argument) for argument in sys.argv[1:5])
BASE = sys.argv[5]

DURATION_ROW = re.compile(r"^([0-9.]+)s call\s+(\S+)$")
durations = {}
for line in DURATIONS.read_text().splitlines():
    match = DURATION_ROW.match(line)
    if match:
        durations[match.group(2)] = float(match.group(1))

S, M, F, E = "STAY-SINGLE", "FOLD-MACHINE", "FUZZ", "MERGE-EXISTING"
PIN = "pin verbatim"

C = {}
def put(names, kind, target, flag=""):
    for n in names.split():
        C[n] = (kind, target, flag)

# basis/test_basis.py
put("test_basis", E, "BasisLifecycle pins; legacy end-to-end in an EXEMPT module, numerics anchored in test_fourier_basis/test_coeff_anchors", "literal modes=150 resize pinned as @initialize case")
# basis/test_coeff_anchors.py
put("test_resize_modes_truncate_values test_resize_modes_pad_values", F, "fuzz source/target counts with an index-copy oracle; current hand literal kept as @example", "ORACLE RISK: hand-computed analytic values hold only at the literal; keep the literal as an analytic pin, fuzz as oracle_independent")
put("test_padding_preserves_conjugate_symmetry test_padding_then_truncating_returns_the_original test_odd_source_counts_have_nothing_to_redistribute test_padding_preserves_the_function_at_the_original_sample_points", M, "ResizeRoundTrips (new): rules pad/truncate with drawn counts; invariants Hermitian, function preserved at original points, odd counts copy through; the loops' enumerated n become @initialize pins", "RESIZE_TOLERANCE is a measured bound: pin verbatim")
put("test_zero_coeff_values", M, "BandLimiting: hand literal as @initialize pin", "analytic literal pinned verbatim")
# basis/test_config.py
put("test_config_resolve_domain_broadcasts_single_pair test_config_lengths test_domain_malformed_pair_raises test_config_with_domain_is_immutable_copy", F, "one fuzzed SpectralConfig test over domain pairs and mode tuples; current literals as @example; malformed pairs from a closed set")
put("test_domain_setter_rebuilds_config", M, "BasisLifecycle rule set_domain; invariant lengths == domain spans")
# basis/test_fourier_basis.py
put("test_transform_requires_2d test_inv_transform_requires_2d test_fn_requires_modes test_fn_dimension_mismatch test_coeff_setter_rejects_non_complex test_call_requires_coeff test_evaluate_time_dependent_requires_t test_generate_invalid_value_type test_add_type_error test_sub_type_error test_plot_out_of_bounds test_plot_coefficients_time_dependent_raises", F, "one refusal table: closed set of refusal kinds enumerated (parametrize), each fuzzed over the shapes/types that trigger it")
put("test_transform_casts_real_to_complex test_inv_transform_casts_real_input test_transform_res_int_and_slice test_fn_casts_integer_input test_fn_transpose_matches_shape test_grid_1d_and_2d test_grid_int_arg test_generate_zero_value_type", F, "fuzz sizes/dims/dtypes; current literals as @example")
put("test_evaluate_accepts_real_coeff", F, "fuzz coefficients; real vs complex-cast agree", "atol=1e-4 literal: measure before keeping")
put("test_dealias_mask_keeps_low_frequencies", F, "fuzz n; mask == |k| <= 2n/3 holds for every n (analytic); n=6 pinned")
put("test_empty_basis_modes_and_len test_modes_ndim_domain_time_size test_len_matches_first_dim test_generate_returns_basis test_generate_empty_shape test_generate_complex_funcs_true test_copy_is_independent test_get_values_res_slice test_get_values_res_tuple test_evaluate_i_n_selection test_evaluate_time_dependent test_get_values_and_grid_shapes test_get_values_basis_eval_1d test_get_values_basis_eval_2d test_to_time_dependent_idempotent_when_already_time_dependent test_to_time_independent_noop_when_not_time_dependent test_time_dependent_grad_time_and_space test_time_dependent_integral_time_and_space test_get_values_time_dependent_inverse_and_basis test_resize_time_dependent_to_basis_interpolates_time", M, "BasisLifecycle (new): rules generate/empty/copy/resize/to_time_dependent/to_time_independent/grad/integral/get_values/evaluate; invariants modes, ndim, len, domain, time flags and output shapes consistent with the drawn state; empty basis and idempotent conversions as @initialize pins")
put("test_plot_1d_real test_plot_1d_complex_both_components test_plot_1d_complex_scatter test_plot_2d_real test_plot_2d_complex_default_component_warns test_plot_2d_complex_scatter test_plot_coefficients_2d test_plot_coefficients_on_axes", F, "enumerate the closed plot matrix (dims x real/complex x component x scatter x axes) by parametrize; the warning case pinned")
put("test_prefered_evaluation_mode", S, "constant class fact, no input range")
put("test_basis_diagonal_hooks_default_to_none", S, "base-class contract fact, no input range")
put("test_math_import_smoke", E, "remove: checks torch.pi, no library behaviour")
# basis/test_nufft.py
put("test_evaluate_switches_to_nufft_above_threshold", F, "fuzz point counts either side of the threshold; current npts pinned", "5e-2 approximation bound: measured? check before fuzzing")
put("test_dense_evaluate_chunk_bounds_memory test_nufft_evaluate_does_not_materialize_dense_matrix", S, "process-memory measurement; fuzzing multiplies noisy allocator readings", "0.5 x dense bytes ratio is a measured bound: pin verbatim")
put("test_nufft_available", S, "environment fact")
put("test_nufft_evaluate_shape", F, "fuzz rows/points; (3, 50) pinned")
# model
put("test_SpectralSVR", E, "legacy end-to-end in an EXEMPT module, numerics anchored in test_spectralsvr_anchors; remove or keep as legacy smoke", "smape atol 96e-3 / 11e-2 are measured: pin verbatim if kept")
put("test_every_kind_of_model_refuses_alike", M, "DirectingAndFitting: @initialize pin enumerating every regressor kind")
put("test_fnn_dump_load_roundtrip test_fnn_dump_only_hyperparams test_fnn_load_wrong_type_raises test_lssvr_dump_load_roundtrip test_lssvr_dump_only_hyperparams test_lssvr_load_type_mismatch_raises", M, "RegressorPersistence (new): rules fit, dump full/hyperparams-only, load as same/other kind; invariants predictions equal after full round trip, untrained after hyperparams-only, refusal on kind mismatch; both kinds enumerated")
put("test_lssvr_predict_before_fit_raises test_lssvr_correlation_before_fit_raises test_lssvr_kernel_property_before_fit_defaults_sigma test_lssvr_correlation_and_p_matrix", M, "LSSVRLifecycle (new): unfitted->fitted states; rules predict/correlation/kernel access before and after fit; invariants refusal before fit, shapes after")
put("test_lssvr_invalid_kernel_raises", F, "fuzz unknown kernel names against the closed registry; current name pinned")
put("test_torch_json_encoder_rejects_unknown_type", F, "fuzz object types outside the encodable set; object() pinned")
put("test_lssvr_keeps_a_stated_host_under_an_accelerator", E, "test_lssvr_follows_the_process_default: add the stated-host case to its parametrize")
put("test_inverse_recovers_known_input", F, "fuzz seeds/sizes; current seed pinned", "0.05 relative bound: measure over the fuzzed range first")
put("test_train_rejects_dtype_mismatch test_test_rejects_dtype_mismatch test_forward_rejects_wrong_x_dims test_inverse_before_train_raises test_inverse_features_none_when_regressor_prefitted test_boundary_refuses_a_regressor_that_would_narrow_coefficients test_boundary_allows_a_wider_regressor_but_reports_it test_boundary_refuses_on_every_path_that_reaches_the_regressor test_boundary_is_silent_when_the_precisions_match test_double_precision_survives_the_model_boundary", M, "SpectralSVRBoundary (new): rules build model with drawn basis/regressor precisions, train/test/forward/inverse in any order; invariants narrowing refused on every path, widening reported, matching silent, inverse refused before train, double precision preserved end to end")
put("test_bridge_round_trips_exactly_at_both_precisions", E, "test_utils::test_every_conversion_preserves_precision (once fuzzed): add exact round-trip at both precisions as an assertion there")
put("test_the_inverse_builds_its_own_source_where_the_work_is test_the_inverse_refuses_a_source_placed_elsewhere", M, "DirectingAndFitting: rules inverse with built / placed random source")
# operators
put("test_provenance_reports_a_dirty_tree_and_a_clean_one", F, "fuzz the kind of dirtiness (modified, untracked, staged, none) over a fixture repo; the clean->dirty pair pinned")
put("test_one_recipe_regenerates_the_record_with_every_default_stated", S, "Justfile fact")
put("test_quick_builds_a_smaller_case_than_full", M, "BenchmarkCasesStayPut: rule build quick vs full at a drawn seed; invariant quick sizes below full")
put("test_the_forced_case_sizes_are_stated_by_name", S, "signature fact (keyword-only)")
put("test_every_registered_dataset_is_named_once test_every_forcing_to_solution_row_says_how_its_forcing_was_obtained", S, "registry fact")
put("test_the_deeponet_value_types_are_immutable test_the_value_types_are_immutable", F, "one enumerated immutability test over the closed set of value types (parametrize), merging both modules' lists")
put("test_operator_dataset_shapes", M, "SensorsAndDatasets: shapes as invariant of build_a_dataset")
put("test_spectralsvr_operator_learns_antiderivative", F, "fuzz seeds/sizes; current seed pinned", "CLOSED_FORM_TOLERANCE=0.70 is measured over seeds 0-9: pin verbatim, remeasure over the fuzzed range")
put("test_an_unfitted_operator_cannot_predict", S, "type-level protocol fact on one adapter; could enumerate both adapters")
put("test_benchmark_returns_metric_table test_benchmark_propagates_a_failing_operator test_benchmark_scores_two_operators_against_the_same_targets", M, "BenchmarkRuns: rules add SpectralSVR/Constant/Exploding operator; invariants one row per operator, shared targets, an exploding fit propagates")
put("test_the_benchmark_script_states_what_its_table_establishes test_benchmark_documents_its_measurement_protocol", S, "docstring text fact")
put("test_a_missing_operator_extra_refuses_by_default", F, "enumerate the closed set (requested operators x extra present x allow-missing) by parametrize")
put("test_fit_seconds_measures_the_fit_and_not_the_prediction", S, "wall-clock timing with sleeps; fuzzing multiplies sleep cost and timer noise", "0.25 s pauses: pin")
put("test_the_prediction_bound_sits_within_ten_times_the_recorded_worst", S, "recorded-measurement fact")
# problems
put("test_axis_diff_rejects_bad_order", E, "test_axis_diff_matches_analytic: add refused orders as a fuzzed refusal over integers outside {1, 2}, 0 and 3 pinned")
put("test_antiderivative_rejects_bad_sizes test_generate_rejects_a_time_domain_without_a_step", F, "fuzz the refused sizes / one-point domains; current literals pinned")
put("test_burgers_numerical_solution_dissipates_energy", F, "fuzz seeds/nu/modes in the unforced regime; current literal pinned", "1e-6 slack literal: measure")
put("test_manufactured_branch_rejects_a_solver test_numerical_branch_uses_the_supplied_solver", F, "enumerate the closed set (branch x solver given / omitted) by parametrize")
put("test_numerical_burgers_agrees_with_an_independent_integrator", F, "fuzz seeds inside the non-stiff regime the test states; current literal pinned", "1e-4 agreement and regime boundary are measured: pin verbatim")
put("test_manufactured_pair_satisfies_the_continuous_equation test_continuous_residual_detects_a_corrupted_forcing", F, "fuzz seeds/bands; current case pinned", "MANUFACTURED_RESIDUAL_TOLERANCE measured: pin verbatim")
put("test_numerical_burgers_reproduces_manufactured_solution test_round_trip_is_not_satisfied_by_a_solver_that_stands_still test_numerical_burgers_round_trip_converges_at_the_integrator_order test_numerical_burgers_forcing_frozen_at_step_start_loses_the_order", S, "convergence studies over a fixed step sweep with recorded constants (travel, orders, standing-still error); fuzzing changes the measured values", "measured constants MANUFACTURED_ROUND_TRIP_*, FROZEN_FORCING_ORDERS, STANDING_STILL_WORST_ERROR: pin verbatim")
put("test_generation_survives_a_process_default_pointing_elsewhere", F, "enumerate DEVICES_PRESENT x process default (closed set) by parametrize")
# guards
put("test_every_module_has_an_anchor test_exempt_list_is_current test_mms_markers_are_retired test_oracle_marker_names_match_conftest", S, "repository guard")
put("test_setup_enforcement_rejects_unmarked_test test_setup_enforcement_rejects_multiple_oracle_markers test_setup_enforcement_accepts_single_oracle_marker", F, "one pytester test enumerating marker sets (none, one each, two) by parametrize")
put("test_no_file_carries_more_prose_than_its_budget test_no_budget_holds_slack test_every_ruff_docstring_finding_lands_in_a_counted_docstring", S, "repository guard")
put("test_a_cyclic_reexport_between_packages_terminates", S, "one fixed malformed package; Python cannot import it, so the fuzzed export test cannot host it")
put("test_scenario_names_are_unique test_every_stated_scenario_is_covered_or_exempt test_no_exemption_is_stale test_every_in_flight_exemption_names_a_scenario_still_in_flight", S, "repository guard")
put("test_the_forms_that_promise_a_real_result_are_exactly_two test_exported_from_the_package_root", S, "type/export fact")
put("test_the_raw_form_states_the_guarantee_it_does_not_make", S, "docstring text fact")
# utils
put("test_get_metrics_hand_computed_values", F, "fuzz arrays against independent numpy formulas; the hand literal kept as an analytic @example", "ORACLE RISK: analytic only at the literal; keep it pinned")
put("test_get_metrics_rrse_near_constant_column", F, "fuzz near-constant columns; the regression literal pinned", "regression case: pin verbatim")
put("test_get_metrics_rse_consistent_with_definition_rrse_squared test_get_metrics_single_sample_r2_is_nan", E, "test_get_metrics_default_keys: add rrse**2 == rse and r2 NaN at one row as assertions/@example")
put("test_get_metrics_nan_count test_get_metrics_injected_metric_set", F, "fuzz NaN positions / injected metric names; literals pinned")
put("test_etdrk4_order_with_time_dependent_forcing", S, "convergence sweep with a recorded order bound", "order > 3.9: pin verbatim")
put("test_etdrk4_complex_weights_match_closed_form test_etdrk4_real_weights_unchanged", F, "fuzz eigenvalues/step sizes; current case pinned", "1e-10 closed-form agreement: measure over range")
put("test_etdrk4_rejects_non_uniform_grid test_etdrk4_requires_matching_shapes test_euler_solver_requires_multiple_points", F, "one solver refusal table, enumerated kinds, fuzzed triggering inputs")
put("test_conversion_integrates_complex_spectral_state", E, "test_conversion_equals_the_combined_right_hand_side: add complex state as @example")
put("test_conversion_forwards_the_time_grid_unchanged", F, "fuzz grids; identity of forwarded objects asserted per draw")
put("test_accelerator_names_whichever_one_is_present test_a_stated_host_survives_an_available_accelerator test_two_stated_indices_that_differ_are_a_disagreement test_an_unstated_index_agrees_with_a_stated_one test_a_random_source_device_is_invisible_to_a_type_checker", M, "DeviceStatements (new): rules state host/index/unstated, repoint default, make a generator; invariants resolve/name/refuse/agree per statement")
put("test_standard_scaler_tuple_and_subset test_standard_scaler_consistency_checks test_standard_scaler_save_load", M, "ScalerLifecycle (new): rules fit, transform tuple/subset, save/load, misuse; invariants inverse(transform) identity, loaded equals original, misuse refused")
put("test_to_mag_angle_idempotent_on_real test_from_mag_angle_idempotent_on_complex", F, "fuzz arrays over the existing real_arrays/complex strategies; current literals pinned")
put("test_to_mag_angle_handles_any_number_of_mode_axes test_from_mag_angle_inverts_to_mag_angle test_polar_encoding_precision_corresponds test_polar_encoding_preserves_device", E, "test_every_conversion_* (polar is in the registry): duplicates; remove after those are fuzzed")
put("test_zero_coeff_per_axis_bands test_zero_coeff_rejects_a_bare_count", M, "BandLimiting: per-axis bands as a rule, bare count as a refusal pin")
put("test_conversion_registry_covers_every_exported_conversion", S, "registry fact")
put("test_every_conversion_accepts_any_number_of_mode_axes test_every_conversion_preserves_precision test_every_conversion_preserves_device", F, "replace internal loops with @given over shapes x dtypes, registry entries enumerated by parametrize")

def located(nodeid):
    file, name = nodeid.split("::")
    tree = ast.parse((ROOT / file).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    oracles = [
        d.attr
        for decorator in node.decorator_list
        for d in [decorator.func if isinstance(decorator, ast.Call) else decorator]
        if isinstance(d, ast.Attribute) and d.attr.startswith("oracle_")
    ]
    return file, name, node.lineno, ",".join(oracles)


singles = json.loads(SINGLES.read_text())
rows = []
unclassified = [nodeid for nodeid in singles if nodeid.split("::")[1] not in C]
stale = sorted(set(C) - {nodeid.split("::")[1] for nodeid in singles})
assert not unclassified and not stale, (unclassified, stale)
for nodeid, scenarios in singles.items():
    file, name, lineno, oracle = located(nodeid)
    kind, target, flag = C[name]
    rows.append((file.removeprefix("tests/"), name, str(lineno), oracle or "(module mark)", ";".join(scenarios), kind, target, flag, durations.get(nodeid, 0.0)))

kinds = Counter(r[5] for r in rows)
per_module = defaultdict(Counter)
for r in rows:
    per_module[r[0]][r[5]] += 1
machines = defaultdict(list)
for r in rows:
    if r[5] == M:
        machines[r[6].split(":")[0].split(" (")[0]].append(r[1])

lines = ["# Consolidation inventory: single tests", "",
         f"{len(rows)} single tests (no @given, no parametrize, not a state machine) at {BASE}.", "",
         "## Counts", "", "| classification | count |", "| --- | ---: |"]
lines += [f"| {k} | {v} |" for k, v in kinds.most_common()]
lines += ["", "| module | STAY-SINGLE | FOLD-MACHINE | FUZZ | MERGE-EXISTING |", "| --- | ---: | ---: | ---: | ---: |"]
for module in sorted(per_module):
    c = per_module[module]
    lines.append(f"| {module} | {c[S]} | {c[M]} | {c[F]} | {c[E]} |")
lines += ["", "## Machines", ""]
for machine, members in sorted(machines.items(), key=lambda kv: -len(kv[1])):
    target = next(r[6] for r in rows if r[5] == M and r[6].startswith(machine))
    lines += [f"### {machine} ({len(members)})", "", target, "", "Members: " + ", ".join(f"`{m}`" for m in members), ""]
lines += ["## Inventory", "", "| module | test | line | oracle | scenarios | classification | target / how | flag | s |", "| --- | --- | ---: | --- | --- | --- | --- | --- | ---: |"]
for r in rows:
    lines.append("| " + " | ".join([r[0], f"`{r[1]}`", r[2], r[3], r[4], r[5], r[6].replace("|", "/"), r[7], f"{r[8]:.2f}"]) + " |")
OUT.write_text("\n".join(lines) + "\n")
print(dict(kinds))
for module in sorted(per_module): print(module, dict(per_module[module]))
for machine, members in sorted(machines.items(), key=lambda kv: -len(kv[1])): print(machine, len(members))
print("flags:", sum(1 for r in rows if r[7]))
