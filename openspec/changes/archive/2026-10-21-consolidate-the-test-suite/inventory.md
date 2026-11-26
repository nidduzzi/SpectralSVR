# Consolidation inventory: single tests

185 single tests (no @given, no parametrize, not a state machine) at 86bbb1c.

## Counts

| classification | count |
| --- | ---: |
| FUZZ | 73 |
| FOLD-MACHINE | 64 |
| STAY-SINGLE | 35 |
| MERGE-EXISTING | 13 |

| module | STAY-SINGLE | FOLD-MACHINE | FUZZ | MERGE-EXISTING |
| --- | ---: | ---: | ---: | ---: |
| basis/test_basis.py | 0 | 0 | 0 | 1 |
| basis/test_coeff_anchors.py | 0 | 5 | 2 | 0 |
| basis/test_config.py | 0 | 1 | 4 | 0 |
| basis/test_fourier_basis.py | 2 | 20 | 30 | 1 |
| basis/test_nufft.py | 3 | 0 | 2 | 0 |
| model/test_SpectralSVR.py | 0 | 0 | 0 | 1 |
| model/test_fitted_placement.py | 0 | 1 | 0 | 0 |
| model/test_fnn.py | 0 | 3 | 0 | 0 |
| model/test_lssvr.py | 0 | 7 | 2 | 1 |
| model/test_spectralsvr_anchors.py | 0 | 0 | 1 | 0 |
| model/test_spectralsvr_methods.py | 0 | 12 | 0 | 1 |
| operators/test_benchmark_record.py | 1 | 0 | 1 | 0 |
| operators/test_burgers_forced_case.py | 3 | 1 | 0 | 0 |
| operators/test_deeponet.py | 0 | 0 | 1 | 0 |
| operators/test_operators.py | 4 | 4 | 3 | 0 |
| operators/test_time_varying_scoring.py | 1 | 0 | 0 | 0 |
| problems/test_problems.py | 4 | 0 | 9 | 1 |
| test_oracle_coverage.py | 4 | 0 | 3 | 0 |
| test_prose_budget.py | 4 | 0 | 0 | 0 |
| test_scenario_coverage.py | 4 | 0 | 0 | 0 |
| utils/test_band.py | 3 | 0 | 0 | 0 |
| utils/test_metric_anchors.py | 0 | 0 | 3 | 1 |
| utils/test_solvers.py | 1 | 0 | 5 | 1 |
| utils/test_utils.py | 1 | 10 | 7 | 5 |

## Machines

### BasisLifecycle (20)

BasisLifecycle rule set_domain; invariant lengths == domain spans

Members: `test_empty_basis_modes_and_len`, `test_modes_ndim_domain_time_size`, `test_len_matches_first_dim`, `test_get_values_res_slice`, `test_get_values_res_tuple`, `test_evaluate_i_n_selection`, `test_evaluate_time_dependent`, `test_generate_returns_basis`, `test_generate_empty_shape`, `test_generate_complex_funcs_true`, `test_copy_is_independent`, `test_get_values_and_grid_shapes`, `test_get_values_basis_eval_1d`, `test_get_values_basis_eval_2d`, `test_to_time_dependent_idempotent_when_already_time_dependent`, `test_to_time_independent_noop_when_not_time_dependent`, `test_time_dependent_grad_time_and_space`, `test_time_dependent_integral_time_and_space`, `test_get_values_time_dependent_inverse_and_basis`, `test_resize_time_dependent_to_basis_interpolates_time`

### SpectralSVRBoundary (10)

SpectralSVRBoundary (new): rules build model with drawn basis/regressor precisions, train/test/forward/inverse in any order; invariants narrowing refused on every path, widening reported, matching silent, inverse refused before train, double precision preserved end to end

Members: `test_train_rejects_dtype_mismatch`, `test_test_rejects_dtype_mismatch`, `test_forward_rejects_wrong_x_dims`, `test_inverse_before_train_raises`, `test_inverse_features_none_when_regressor_prefitted`, `test_boundary_refuses_a_regressor_that_would_narrow_coefficients`, `test_boundary_allows_a_wider_regressor_but_reports_it`, `test_boundary_refuses_on_every_path_that_reaches_the_regressor`, `test_boundary_is_silent_when_the_precisions_match`, `test_double_precision_survives_the_model_boundary`

### RegressorPersistence (6)

RegressorPersistence (new): rules fit, dump full/hyperparams-only, load as same/other kind; invariants predictions equal after full round trip, untrained after hyperparams-only, refusal on kind mismatch; both kinds enumerated

Members: `test_fnn_dump_load_roundtrip`, `test_fnn_dump_only_hyperparams`, `test_fnn_load_wrong_type_raises`, `test_lssvr_dump_load_roundtrip`, `test_lssvr_dump_only_hyperparams`, `test_lssvr_load_type_mismatch_raises`

### DeviceStatements (5)

DeviceStatements (new): rules state host/index/unstated, repoint default, make a generator; invariants resolve/name/refuse/agree per statement

Members: `test_accelerator_names_whichever_one_is_present`, `test_a_stated_host_survives_an_available_accelerator`, `test_two_stated_indices_that_differ_are_a_disagreement`, `test_an_unstated_index_agrees_with_a_stated_one`, `test_a_random_source_device_is_invisible_to_a_type_checker`

### ResizeRoundTrips (4)

ResizeRoundTrips (new): rules pad/truncate with drawn counts; invariants Hermitian, function preserved at original points, odd counts copy through; the loops' enumerated n become @initialize pins

Members: `test_padding_preserves_conjugate_symmetry`, `test_padding_then_truncating_returns_the_original`, `test_odd_source_counts_have_nothing_to_redistribute`, `test_padding_preserves_the_function_at_the_original_sample_points`

### LSSVRLifecycle (4)

LSSVRLifecycle (new): unfitted->fitted states; rules predict/correlation/kernel access before and after fit; invariants refusal before fit, shapes after

Members: `test_lssvr_predict_before_fit_raises`, `test_lssvr_correlation_and_p_matrix`, `test_lssvr_correlation_before_fit_raises`, `test_lssvr_kernel_property_before_fit_defaults_sigma`

### BandLimiting (3)

BandLimiting: hand literal as @initialize pin

Members: `test_zero_coeff_values`, `test_zero_coeff_per_axis_bands`, `test_zero_coeff_rejects_a_bare_count`

### DirectingAndFitting (3)

DirectingAndFitting: @initialize pin enumerating every regressor kind

Members: `test_every_kind_of_model_refuses_alike`, `test_the_inverse_builds_its_own_source_where_the_work_is`, `test_the_inverse_refuses_a_source_placed_elsewhere`

### BenchmarkRuns (3)

BenchmarkRuns: rules add SpectralSVR/Constant/Exploding operator; invariants one row per operator, shared targets, an exploding fit propagates

Members: `test_benchmark_returns_metric_table`, `test_benchmark_propagates_a_failing_operator`, `test_benchmark_scores_two_operators_against_the_same_targets`

### ScalerLifecycle (3)

ScalerLifecycle (new): rules fit, transform tuple/subset, save/load, misuse; invariants inverse(transform) identity, loaded equals original, misuse refused

Members: `test_standard_scaler_tuple_and_subset`, `test_standard_scaler_consistency_checks`, `test_standard_scaler_save_load`

### BasisLifecycle rule set_domain; invariant lengths == domain spans (1)

BasisLifecycle rule set_domain; invariant lengths == domain spans

Members: `test_domain_setter_rebuilds_config`

### BenchmarkCasesStayPut (1)

BenchmarkCasesStayPut: rule build quick vs full at a drawn seed; invariant quick sizes below full

Members: `test_quick_builds_a_smaller_case_than_full`

### SensorsAndDatasets (1)

SensorsAndDatasets: shapes as invariant of build_a_dataset

Members: `test_operator_dataset_shapes`

## Inventory

| module | test | line | oracle | scenarios | classification | target / how | flag | s |
| --- | --- | ---: | --- | --- | --- | --- | --- | ---: |
| basis/test_basis.py | `test_basis` | 11 | (module mark) |  | MERGE-EXISTING | BasisLifecycle pins; legacy end-to-end in an EXEMPT module, numerics anchored in test_fourier_basis/test_coeff_anchors | literal modes=150 resize pinned as @initialize case | 0.04 |
| basis/test_coeff_anchors.py | `test_resize_modes_truncate_values` | 41 | oracle_analytic |  | FUZZ | fuzz source/target counts with an index-copy oracle; current hand literal kept as @example | ORACLE RISK: hand-computed analytic values hold only at the literal; keep the literal as an analytic pin, fuzz as oracle_independent | 0.00 |
| basis/test_coeff_anchors.py | `test_resize_modes_pad_values` | 56 | oracle_analytic |  | FUZZ | fuzz source/target counts with an index-copy oracle; current hand literal kept as @example | ORACLE RISK: hand-computed analytic values hold only at the literal; keep the literal as an analytic pin, fuzz as oracle_independent | 0.00 |
| basis/test_coeff_anchors.py | `test_padding_preserves_conjugate_symmetry` | 84 | oracle_invariant |  | FOLD-MACHINE | ResizeRoundTrips (new): rules pad/truncate with drawn counts; invariants Hermitian, function preserved at original points, odd counts copy through; the loops' enumerated n become @initialize pins | RESIZE_TOLERANCE is a measured bound: pin verbatim | 0.00 |
| basis/test_coeff_anchors.py | `test_padding_then_truncating_returns_the_original` | 95 | oracle_invariant |  | FOLD-MACHINE | ResizeRoundTrips (new): rules pad/truncate with drawn counts; invariants Hermitian, function preserved at original points, odd counts copy through; the loops' enumerated n become @initialize pins | RESIZE_TOLERANCE is a measured bound: pin verbatim | 0.00 |
| basis/test_coeff_anchors.py | `test_odd_source_counts_have_nothing_to_redistribute` | 110 | oracle_invariant |  | FOLD-MACHINE | ResizeRoundTrips (new): rules pad/truncate with drawn counts; invariants Hermitian, function preserved at original points, odd counts copy through; the loops' enumerated n become @initialize pins | RESIZE_TOLERANCE is a measured bound: pin verbatim | 0.00 |
| basis/test_coeff_anchors.py | `test_padding_preserves_the_function_at_the_original_sample_points` | 125 | oracle_invariant |  | FOLD-MACHINE | ResizeRoundTrips (new): rules pad/truncate with drawn counts; invariants Hermitian, function preserved at original points, odd counts copy through; the loops' enumerated n become @initialize pins | RESIZE_TOLERANCE is a measured bound: pin verbatim | 0.00 |
| basis/test_coeff_anchors.py | `test_zero_coeff_values` | 137 | oracle_analytic |  | FOLD-MACHINE | BandLimiting: hand literal as @initialize pin | analytic literal pinned verbatim | 0.00 |
| basis/test_config.py | `test_config_resolve_domain_broadcasts_single_pair` | 14 | oracle_smoke |  | FUZZ | one fuzzed SpectralConfig test over domain pairs and mode tuples; current literals as @example; malformed pairs from a closed set |  | 0.00 |
| basis/test_config.py | `test_config_lengths` | 21 | oracle_smoke |  | FUZZ | one fuzzed SpectralConfig test over domain pairs and mode tuples; current literals as @example; malformed pairs from a closed set |  | 0.00 |
| basis/test_config.py | `test_config_with_domain_is_immutable_copy` | 28 | oracle_smoke |  | FUZZ | one fuzzed SpectralConfig test over domain pairs and mode tuples; current literals as @example; malformed pairs from a closed set |  | 0.00 |
| basis/test_config.py | `test_domain_setter_rebuilds_config` | 39 | oracle_smoke |  | FOLD-MACHINE | BasisLifecycle rule set_domain; invariant lengths == domain spans |  | 0.00 |
| basis/test_config.py | `test_domain_malformed_pair_raises` | 48 | oracle_smoke |  | FUZZ | one fuzzed SpectralConfig test over domain pairs and mode tuples; current literals as @example; malformed pairs from a closed set |  | 0.00 |
| basis/test_fourier_basis.py | `test_transform_requires_2d` | 423 | oracle_smoke |  | FUZZ | one refusal table: closed set of refusal kinds enumerated (parametrize), each fuzzed over the shapes/types that trigger it |  | 0.00 |
| basis/test_fourier_basis.py | `test_inv_transform_requires_2d` | 430 | oracle_smoke |  | FUZZ | one refusal table: closed set of refusal kinds enumerated (parametrize), each fuzzed over the shapes/types that trigger it |  | 0.00 |
| basis/test_fourier_basis.py | `test_transform_casts_real_to_complex` | 437 | oracle_smoke |  | FUZZ | fuzz sizes/dims/dtypes; current literals as @example |  | 0.00 |
| basis/test_fourier_basis.py | `test_inv_transform_casts_real_input` | 443 | oracle_smoke |  | FUZZ | fuzz sizes/dims/dtypes; current literals as @example |  | 0.00 |
| basis/test_fourier_basis.py | `test_transform_res_int_and_slice` | 449 | oracle_smoke |  | FUZZ | fuzz sizes/dims/dtypes; current literals as @example |  | 0.00 |
| basis/test_fourier_basis.py | `test_fn_requires_modes` | 457 | oracle_smoke |  | FUZZ | one refusal table: closed set of refusal kinds enumerated (parametrize), each fuzzed over the shapes/types that trigger it |  | 0.00 |
| basis/test_fourier_basis.py | `test_fn_dimension_mismatch` | 464 | oracle_smoke |  | FUZZ | one refusal table: closed set of refusal kinds enumerated (parametrize), each fuzzed over the shapes/types that trigger it |  | 0.00 |
| basis/test_fourier_basis.py | `test_fn_casts_integer_input` | 471 | oracle_smoke |  | FUZZ | fuzz sizes/dims/dtypes; current literals as @example |  | 0.00 |
| basis/test_fourier_basis.py | `test_fn_transpose_matches_shape` | 478 | oracle_smoke |  | FUZZ | fuzz sizes/dims/dtypes; current literals as @example |  | 0.00 |
| basis/test_fourier_basis.py | `test_coeff_setter_rejects_non_complex` | 485 | oracle_smoke |  | FUZZ | one refusal table: closed set of refusal kinds enumerated (parametrize), each fuzzed over the shapes/types that trigger it |  | 0.00 |
| basis/test_fourier_basis.py | `test_empty_basis_modes_and_len` | 492 | oracle_smoke |  | FOLD-MACHINE | BasisLifecycle (new): rules generate/empty/copy/resize/to_time_dependent/to_time_independent/grad/integral/get_values/evaluate; invariants modes, ndim, len, domain, time flags and output shapes consistent with the drawn state; empty basis and idempotent conversions as @initialize pins |  | 0.00 |
| basis/test_fourier_basis.py | `test_modes_ndim_domain_time_size` | 502 | oracle_smoke |  | FOLD-MACHINE | BasisLifecycle (new): rules generate/empty/copy/resize/to_time_dependent/to_time_independent/grad/integral/get_values/evaluate; invariants modes, ndim, len, domain, time flags and output shapes consistent with the drawn state; empty basis and idempotent conversions as @initialize pins |  | 0.00 |
| basis/test_fourier_basis.py | `test_len_matches_first_dim` | 515 | oracle_smoke |  | FOLD-MACHINE | BasisLifecycle (new): rules generate/empty/copy/resize/to_time_dependent/to_time_independent/grad/integral/get_values/evaluate; invariants modes, ndim, len, domain, time flags and output shapes consistent with the drawn state; empty basis and idempotent conversions as @initialize pins |  | 0.00 |
| basis/test_fourier_basis.py | `test_call_requires_coeff` | 521 | oracle_smoke |  | FUZZ | one refusal table: closed set of refusal kinds enumerated (parametrize), each fuzzed over the shapes/types that trigger it |  | 0.00 |
| basis/test_fourier_basis.py | `test_evaluate_accepts_real_coeff` | 528 | oracle_consistency |  | FUZZ | fuzz coefficients; real vs complex-cast agree | atol=1e-4 literal: measure before keeping | 0.00 |
| basis/test_fourier_basis.py | `test_get_values_res_slice` | 538 | oracle_smoke |  | FOLD-MACHINE | BasisLifecycle (new): rules generate/empty/copy/resize/to_time_dependent/to_time_independent/grad/integral/get_values/evaluate; invariants modes, ndim, len, domain, time flags and output shapes consistent with the drawn state; empty basis and idempotent conversions as @initialize pins |  | 0.00 |
| basis/test_fourier_basis.py | `test_get_values_res_tuple` | 546 | oracle_smoke |  | FOLD-MACHINE | BasisLifecycle (new): rules generate/empty/copy/resize/to_time_dependent/to_time_independent/grad/integral/get_values/evaluate; invariants modes, ndim, len, domain, time flags and output shapes consistent with the drawn state; empty basis and idempotent conversions as @initialize pins |  | 0.00 |
| basis/test_fourier_basis.py | `test_evaluate_i_n_selection` | 554 | oracle_smoke |  | FOLD-MACHINE | BasisLifecycle (new): rules generate/empty/copy/resize/to_time_dependent/to_time_independent/grad/integral/get_values/evaluate; invariants modes, ndim, len, domain, time flags and output shapes consistent with the drawn state; empty basis and idempotent conversions as @initialize pins |  | 0.00 |
| basis/test_fourier_basis.py | `test_evaluate_time_dependent` | 562 | oracle_smoke |  | FOLD-MACHINE | BasisLifecycle (new): rules generate/empty/copy/resize/to_time_dependent/to_time_independent/grad/integral/get_values/evaluate; invariants modes, ndim, len, domain, time flags and output shapes consistent with the drawn state; empty basis and idempotent conversions as @initialize pins |  | 0.00 |
| basis/test_fourier_basis.py | `test_evaluate_time_dependent_requires_t` | 576 | oracle_smoke |  | FUZZ | one refusal table: closed set of refusal kinds enumerated (parametrize), each fuzzed over the shapes/types that trigger it |  | 0.00 |
| basis/test_fourier_basis.py | `test_generate_returns_basis` | 589 | oracle_smoke |  | FOLD-MACHINE | BasisLifecycle (new): rules generate/empty/copy/resize/to_time_dependent/to_time_independent/grad/integral/get_values/evaluate; invariants modes, ndim, len, domain, time flags and output shapes consistent with the drawn state; empty basis and idempotent conversions as @initialize pins |  | 0.00 |
| basis/test_fourier_basis.py | `test_generate_zero_value_type` | 596 | oracle_smoke |  | FUZZ | fuzz sizes/dims/dtypes; current literals as @example |  | 0.00 |
| basis/test_fourier_basis.py | `test_generate_empty_shape` | 604 | oracle_smoke |  | FOLD-MACHINE | BasisLifecycle (new): rules generate/empty/copy/resize/to_time_dependent/to_time_independent/grad/integral/get_values/evaluate; invariants modes, ndim, len, domain, time flags and output shapes consistent with the drawn state; empty basis and idempotent conversions as @initialize pins |  | 0.00 |
| basis/test_fourier_basis.py | `test_generate_complex_funcs_true` | 612 | oracle_smoke |  | FOLD-MACHINE | BasisLifecycle (new): rules generate/empty/copy/resize/to_time_dependent/to_time_independent/grad/integral/get_values/evaluate; invariants modes, ndim, len, domain, time flags and output shapes consistent with the drawn state; empty basis and idempotent conversions as @initialize pins |  | 0.00 |
| basis/test_fourier_basis.py | `test_generate_invalid_value_type` | 623 | oracle_smoke |  | FUZZ | one refusal table: closed set of refusal kinds enumerated (parametrize), each fuzzed over the shapes/types that trigger it |  | 0.00 |
| basis/test_fourier_basis.py | `test_add_type_error` | 630 | oracle_smoke |  | FUZZ | one refusal table: closed set of refusal kinds enumerated (parametrize), each fuzzed over the shapes/types that trigger it |  | 0.00 |
| basis/test_fourier_basis.py | `test_sub_type_error` | 637 | oracle_smoke |  | FUZZ | one refusal table: closed set of refusal kinds enumerated (parametrize), each fuzzed over the shapes/types that trigger it |  | 0.00 |
| basis/test_fourier_basis.py | `test_copy_is_independent` | 644 | oracle_smoke |  | FOLD-MACHINE | BasisLifecycle (new): rules generate/empty/copy/resize/to_time_dependent/to_time_independent/grad/integral/get_values/evaluate; invariants modes, ndim, len, domain, time flags and output shapes consistent with the drawn state; empty basis and idempotent conversions as @initialize pins |  | 0.00 |
| basis/test_fourier_basis.py | `test_prefered_evaluation_mode` | 654 | oracle_smoke |  | STAY-SINGLE | constant class fact, no input range |  | 0.00 |
| basis/test_fourier_basis.py | `test_grid_1d_and_2d` | 660 | oracle_smoke |  | FUZZ | fuzz sizes/dims/dtypes; current literals as @example |  | 0.00 |
| basis/test_fourier_basis.py | `test_grid_int_arg` | 670 | oracle_smoke |  | FUZZ | fuzz sizes/dims/dtypes; current literals as @example |  | 0.00 |
| basis/test_fourier_basis.py | `test_get_values_and_grid_shapes` | 679 | oracle_smoke |  | FOLD-MACHINE | BasisLifecycle (new): rules generate/empty/copy/resize/to_time_dependent/to_time_independent/grad/integral/get_values/evaluate; invariants modes, ndim, len, domain, time flags and output shapes consistent with the drawn state; empty basis and idempotent conversions as @initialize pins |  | 0.00 |
| basis/test_fourier_basis.py | `test_get_values_basis_eval_1d` | 687 | oracle_smoke |  | FOLD-MACHINE | BasisLifecycle (new): rules generate/empty/copy/resize/to_time_dependent/to_time_independent/grad/integral/get_values/evaluate; invariants modes, ndim, len, domain, time flags and output shapes consistent with the drawn state; empty basis and idempotent conversions as @initialize pins |  | 0.00 |
| basis/test_fourier_basis.py | `test_get_values_basis_eval_2d` | 697 | oracle_smoke |  | FOLD-MACHINE | BasisLifecycle (new): rules generate/empty/copy/resize/to_time_dependent/to_time_independent/grad/integral/get_values/evaluate; invariants modes, ndim, len, domain, time flags and output shapes consistent with the drawn state; empty basis and idempotent conversions as @initialize pins |  | 0.00 |
| basis/test_fourier_basis.py | `test_to_time_dependent_idempotent_when_already_time_dependent` | 764 | oracle_smoke |  | FOLD-MACHINE | BasisLifecycle (new): rules generate/empty/copy/resize/to_time_dependent/to_time_independent/grad/integral/get_values/evaluate; invariants modes, ndim, len, domain, time flags and output shapes consistent with the drawn state; empty basis and idempotent conversions as @initialize pins |  | 0.00 |
| basis/test_fourier_basis.py | `test_to_time_independent_noop_when_not_time_dependent` | 772 | oracle_smoke |  | FOLD-MACHINE | BasisLifecycle (new): rules generate/empty/copy/resize/to_time_dependent/to_time_independent/grad/integral/get_values/evaluate; invariants modes, ndim, len, domain, time flags and output shapes consistent with the drawn state; empty basis and idempotent conversions as @initialize pins |  | 0.00 |
| basis/test_fourier_basis.py | `test_time_dependent_grad_time_and_space` | 780 | oracle_smoke |  | FOLD-MACHINE | BasisLifecycle (new): rules generate/empty/copy/resize/to_time_dependent/to_time_independent/grad/integral/get_values/evaluate; invariants modes, ndim, len, domain, time flags and output shapes consistent with the drawn state; empty basis and idempotent conversions as @initialize pins |  | 0.00 |
| basis/test_fourier_basis.py | `test_time_dependent_integral_time_and_space` | 788 | oracle_smoke |  | FOLD-MACHINE | BasisLifecycle (new): rules generate/empty/copy/resize/to_time_dependent/to_time_independent/grad/integral/get_values/evaluate; invariants modes, ndim, len, domain, time flags and output shapes consistent with the drawn state; empty basis and idempotent conversions as @initialize pins |  | 0.00 |
| basis/test_fourier_basis.py | `test_get_values_time_dependent_inverse_and_basis` | 796 | oracle_smoke |  | FOLD-MACHINE | BasisLifecycle (new): rules generate/empty/copy/resize/to_time_dependent/to_time_independent/grad/integral/get_values/evaluate; invariants modes, ndim, len, domain, time flags and output shapes consistent with the drawn state; empty basis and idempotent conversions as @initialize pins |  | 0.00 |
| basis/test_fourier_basis.py | `test_resize_time_dependent_to_basis_interpolates_time` | 807 | oracle_smoke |  | FOLD-MACHINE | BasisLifecycle (new): rules generate/empty/copy/resize/to_time_dependent/to_time_independent/grad/integral/get_values/evaluate; invariants modes, ndim, len, domain, time flags and output shapes consistent with the drawn state; empty basis and idempotent conversions as @initialize pins |  | 0.01 |
| basis/test_fourier_basis.py | `test_plot_1d_real` | 840 | oracle_smoke |  | FUZZ | enumerate the closed plot matrix (dims x real/complex x component x scatter x axes) by parametrize; the warning case pinned |  | 0.01 |
| basis/test_fourier_basis.py | `test_plot_1d_complex_both_components` | 846 | oracle_smoke |  | FUZZ | enumerate the closed plot matrix (dims x real/complex x component x scatter x axes) by parametrize; the warning case pinned |  | 0.01 |
| basis/test_fourier_basis.py | `test_plot_1d_complex_scatter` | 859 | oracle_smoke |  | FUZZ | enumerate the closed plot matrix (dims x real/complex x component x scatter x axes) by parametrize; the warning case pinned |  | 0.01 |
| basis/test_fourier_basis.py | `test_plot_2d_real` | 865 | oracle_smoke |  | FUZZ | enumerate the closed plot matrix (dims x real/complex x component x scatter x axes) by parametrize; the warning case pinned |  | 0.03 |
| basis/test_fourier_basis.py | `test_plot_2d_complex_default_component_warns` | 872 | oracle_smoke |  | FUZZ | enumerate the closed plot matrix (dims x real/complex x component x scatter x axes) by parametrize; the warning case pinned |  | 0.00 |
| basis/test_fourier_basis.py | `test_plot_2d_complex_scatter` | 888 | oracle_smoke |  | FUZZ | enumerate the closed plot matrix (dims x real/complex x component x scatter x axes) by parametrize; the warning case pinned |  | 0.25 |
| basis/test_fourier_basis.py | `test_plot_out_of_bounds` | 894 | oracle_smoke |  | FUZZ | one refusal table: closed set of refusal kinds enumerated (parametrize), each fuzzed over the shapes/types that trigger it |  | 0.00 |
| basis/test_fourier_basis.py | `test_plot_coefficients_2d` | 908 | oracle_smoke |  | FUZZ | enumerate the closed plot matrix (dims x real/complex x component x scatter x axes) by parametrize; the warning case pinned |  | 0.01 |
| basis/test_fourier_basis.py | `test_plot_coefficients_on_axes` | 915 | oracle_smoke |  | FUZZ | enumerate the closed plot matrix (dims x real/complex x component x scatter x axes) by parametrize; the warning case pinned |  | 0.00 |
| basis/test_fourier_basis.py | `test_plot_coefficients_time_dependent_raises` | 923 | oracle_smoke |  | FUZZ | one refusal table: closed set of refusal kinds enumerated (parametrize), each fuzzed over the shapes/types that trigger it |  | 0.01 |
| basis/test_fourier_basis.py | `test_dealias_mask_keeps_low_frequencies` | 1055 | oracle_analytic |  | FUZZ | fuzz n; mask == /k/ <= 2n/3 holds for every n (analytic); n=6 pinned |  | 0.00 |
| basis/test_fourier_basis.py | `test_basis_diagonal_hooks_default_to_none` | 1116 | oracle_smoke |  | STAY-SINGLE | base-class contract fact, no input range |  | 0.00 |
| basis/test_fourier_basis.py | `test_math_import_smoke` | 1125 | oracle_smoke |  | MERGE-EXISTING | remove: checks torch.pi, no library behaviour |  | 0.00 |
| basis/test_nufft.py | `test_evaluate_switches_to_nufft_above_threshold` | 74 | oracle_independent |  | FUZZ | fuzz point counts either side of the threshold; current npts pinned | 5e-2 approximation bound: measured? check before fuzzing | 0.01 |
| basis/test_nufft.py | `test_dense_evaluate_chunk_bounds_memory` | 116 | oracle_smoke |  | STAY-SINGLE | process-memory measurement; fuzzing multiplies noisy allocator readings | 0.5 x dense bytes ratio is a measured bound: pin verbatim | 0.44 |
| basis/test_nufft.py | `test_nufft_available` | 197 | oracle_smoke |  | STAY-SINGLE | environment fact |  | 0.00 |
| basis/test_nufft.py | `test_nufft_evaluate_shape` | 203 | oracle_smoke |  | FUZZ | fuzz rows/points; (3, 50) pinned |  | 0.01 |
| basis/test_nufft.py | `test_nufft_evaluate_does_not_materialize_dense_matrix` | 213 | oracle_smoke |  | STAY-SINGLE | process-memory measurement; fuzzing multiplies noisy allocator readings | 0.5 x dense bytes ratio is a measured bound: pin verbatim | 0.24 |
| model/test_SpectralSVR.py | `test_SpectralSVR` | 21 | (module mark) |  | MERGE-EXISTING | legacy end-to-end in an EXEMPT module, numerics anchored in test_spectralsvr_anchors; remove or keep as legacy smoke | smape atol 96e-3 / 11e-2 are measured: pin verbatim if kept | 0.10 |
| model/test_fitted_placement.py | `test_every_kind_of_model_refuses_alike` | 60 | oracle_smoke | Every kind of model refuses alike | FOLD-MACHINE | DirectingAndFitting: @initialize pin enumerating every regressor kind |  | 0.00 |
| model/test_fnn.py | `test_fnn_dump_load_roundtrip` | 48 | oracle_consistency |  | FOLD-MACHINE | RegressorPersistence (new): rules fit, dump full/hyperparams-only, load as same/other kind; invariants predictions equal after full round trip, untrained after hyperparams-only, refusal on kind mismatch; both kinds enumerated |  | 0.00 |
| model/test_fnn.py | `test_fnn_dump_only_hyperparams` | 62 | oracle_smoke |  | FOLD-MACHINE | RegressorPersistence (new): rules fit, dump full/hyperparams-only, load as same/other kind; invariants predictions equal after full round trip, untrained after hyperparams-only, refusal on kind mismatch; both kinds enumerated |  | 0.00 |
| model/test_fnn.py | `test_fnn_load_wrong_type_raises` | 75 | oracle_smoke |  | FOLD-MACHINE | RegressorPersistence (new): rules fit, dump full/hyperparams-only, load as same/other kind; invariants predictions equal after full round trip, untrained after hyperparams-only, refusal on kind mismatch; both kinds enumerated |  | 0.00 |
| model/test_lssvr.py | `test_lssvr_dump_load_roundtrip` | 57 | oracle_consistency |  | FOLD-MACHINE | RegressorPersistence (new): rules fit, dump full/hyperparams-only, load as same/other kind; invariants predictions equal after full round trip, untrained after hyperparams-only, refusal on kind mismatch; both kinds enumerated |  | 0.00 |
| model/test_lssvr.py | `test_lssvr_predict_before_fit_raises` | 72 | oracle_smoke |  | FOLD-MACHINE | LSSVRLifecycle (new): unfitted->fitted states; rules predict/correlation/kernel access before and after fit; invariants refusal before fit, shapes after |  | 0.00 |
| model/test_lssvr.py | `test_lssvr_invalid_kernel_raises` | 96 | oracle_smoke |  | FUZZ | fuzz unknown kernel names against the closed registry; current name pinned |  | 0.00 |
| model/test_lssvr.py | `test_lssvr_correlation_and_p_matrix` | 103 | oracle_smoke |  | FOLD-MACHINE | LSSVRLifecycle (new): unfitted->fitted states; rules predict/correlation/kernel access before and after fit; invariants refusal before fit, shapes after |  | 0.00 |
| model/test_lssvr.py | `test_lssvr_correlation_before_fit_raises` | 115 | oracle_smoke |  | FOLD-MACHINE | LSSVRLifecycle (new): unfitted->fitted states; rules predict/correlation/kernel access before and after fit; invariants refusal before fit, shapes after |  | 0.00 |
| model/test_lssvr.py | `test_lssvr_dump_only_hyperparams` | 122 | oracle_smoke |  | FOLD-MACHINE | RegressorPersistence (new): rules fit, dump full/hyperparams-only, load as same/other kind; invariants predictions equal after full round trip, untrained after hyperparams-only, refusal on kind mismatch; both kinds enumerated |  | 0.00 |
| model/test_lssvr.py | `test_lssvr_kernel_property_before_fit_defaults_sigma` | 138 | oracle_smoke |  | FOLD-MACHINE | LSSVRLifecycle (new): unfitted->fitted states; rules predict/correlation/kernel access before and after fit; invariants refusal before fit, shapes after |  | 0.00 |
| model/test_lssvr.py | `test_lssvr_load_type_mismatch_raises` | 163 | oracle_smoke |  | FOLD-MACHINE | RegressorPersistence (new): rules fit, dump full/hyperparams-only, load as same/other kind; invariants predictions equal after full round trip, untrained after hyperparams-only, refusal on kind mismatch; both kinds enumerated |  | 0.00 |
| model/test_lssvr.py | `test_torch_json_encoder_rejects_unknown_type` | 174 | oracle_smoke |  | FUZZ | fuzz object types outside the encodable set; object() pinned |  | 0.00 |
| model/test_lssvr.py | `test_lssvr_keeps_a_stated_host_under_an_accelerator` | 209 | oracle_smoke | The host is not overridden | MERGE-EXISTING | test_lssvr_follows_the_process_default: add the stated-host case to its parametrize |  | 0.00 |
| model/test_spectralsvr_anchors.py | `test_inverse_recovers_known_input` | 81 | oracle_analytic |  | FUZZ | fuzz seeds/sizes; current seed pinned | 0.05 relative bound: measure over the fuzzed range first | 0.16 |
| model/test_spectralsvr_methods.py | `test_train_rejects_dtype_mismatch` | 123 | oracle_smoke |  | FOLD-MACHINE | SpectralSVRBoundary (new): rules build model with drawn basis/regressor precisions, train/test/forward/inverse in any order; invariants narrowing refused on every path, widening reported, matching silent, inverse refused before train, double precision preserved end to end |  | 0.00 |
| model/test_spectralsvr_methods.py | `test_test_rejects_dtype_mismatch` | 131 | oracle_smoke |  | FOLD-MACHINE | SpectralSVRBoundary (new): rules build model with drawn basis/regressor precisions, train/test/forward/inverse in any order; invariants narrowing refused on every path, widening reported, matching silent, inverse refused before train, double precision preserved end to end |  | 0.00 |
| model/test_spectralsvr_methods.py | `test_forward_rejects_wrong_x_dims` | 139 | oracle_smoke |  | FOLD-MACHINE | SpectralSVRBoundary (new): rules build model with drawn basis/regressor precisions, train/test/forward/inverse in any order; invariants narrowing refused on every path, widening reported, matching silent, inverse refused before train, double precision preserved end to end |  | 0.00 |
| model/test_spectralsvr_methods.py | `test_inverse_before_train_raises` | 147 | oracle_smoke |  | FOLD-MACHINE | SpectralSVRBoundary (new): rules build model with drawn basis/regressor precisions, train/test/forward/inverse in any order; invariants narrowing refused on every path, widening reported, matching silent, inverse refused before train, double precision preserved end to end |  | 0.00 |
| model/test_spectralsvr_methods.py | `test_inverse_features_none_when_regressor_prefitted` | 155 | oracle_smoke |  | FOLD-MACHINE | SpectralSVRBoundary (new): rules build model with drawn basis/regressor precisions, train/test/forward/inverse in any order; invariants narrowing refused on every path, widening reported, matching silent, inverse refused before train, double precision preserved end to end |  | 0.00 |
| model/test_spectralsvr_methods.py | `test_boundary_refuses_a_regressor_that_would_narrow_coefficients` | 192 | oracle_invariant |  | FOLD-MACHINE | SpectralSVRBoundary (new): rules build model with drawn basis/regressor precisions, train/test/forward/inverse in any order; invariants narrowing refused on every path, widening reported, matching silent, inverse refused before train, double precision preserved end to end |  | 0.00 |
| model/test_spectralsvr_methods.py | `test_boundary_allows_a_wider_regressor_but_reports_it` | 200 | oracle_invariant |  | FOLD-MACHINE | SpectralSVRBoundary (new): rules build model with drawn basis/regressor precisions, train/test/forward/inverse in any order; invariants narrowing refused on every path, widening reported, matching silent, inverse refused before train, double precision preserved end to end |  | 0.00 |
| model/test_spectralsvr_methods.py | `test_boundary_refuses_on_every_path_that_reaches_the_regressor` | 210 | oracle_invariant |  | FOLD-MACHINE | SpectralSVRBoundary (new): rules build model with drawn basis/regressor precisions, train/test/forward/inverse in any order; invariants narrowing refused on every path, widening reported, matching silent, inverse refused before train, double precision preserved end to end |  | 0.00 |
| model/test_spectralsvr_methods.py | `test_boundary_is_silent_when_the_precisions_match` | 230 | oracle_invariant |  | FOLD-MACHINE | SpectralSVRBoundary (new): rules build model with drawn basis/regressor precisions, train/test/forward/inverse in any order; invariants narrowing refused on every path, widening reported, matching silent, inverse refused before train, double precision preserved end to end |  | 0.00 |
| model/test_spectralsvr_methods.py | `test_double_precision_survives_the_model_boundary` | 239 | oracle_consistency |  | FOLD-MACHINE | SpectralSVRBoundary (new): rules build model with drawn basis/regressor precisions, train/test/forward/inverse in any order; invariants narrowing refused on every path, widening reported, matching silent, inverse refused before train, double precision preserved end to end |  | 0.00 |
| model/test_spectralsvr_methods.py | `test_bridge_round_trips_exactly_at_both_precisions` | 261 | oracle_invariant |  | MERGE-EXISTING | test_utils::test_every_conversion_preserves_precision (once fuzzed): add exact round-trip at both precisions as an assertion there |  | 0.00 |
| model/test_spectralsvr_methods.py | `test_the_inverse_builds_its_own_source_where_the_work_is` | 274 | oracle_smoke | A source the library builds for itself agrees by construction | FOLD-MACHINE | DirectingAndFitting: rules inverse with built / placed random source |  | 0.02 |
| model/test_spectralsvr_methods.py | `test_the_inverse_refuses_a_source_placed_elsewhere` | 287 | oracle_smoke | A mismatched random source names both devices | FOLD-MACHINE | DirectingAndFitting: rules inverse with built / placed random source |  | 0.00 |
| operators/test_benchmark_record.py | `test_provenance_reports_a_dirty_tree_and_a_clean_one` | 259 | oracle_analytic |  | FUZZ | fuzz the kind of dirtiness (modified, untracked, staged, none) over a fixture repo; the clean->dirty pair pinned |  | 0.01 |
| operators/test_benchmark_record.py | `test_one_recipe_regenerates_the_record_with_every_default_stated` | 299 | oracle_smoke | One command regenerates | STAY-SINGLE | Justfile fact |  | 0.00 |
| operators/test_burgers_forced_case.py | `test_quick_builds_a_smaller_case_than_full` | 206 | oracle_smoke |  | FOLD-MACHINE | BenchmarkCasesStayPut: rule build quick vs full at a drawn seed; invariant quick sizes below full |  | 0.33 |
| operators/test_burgers_forced_case.py | `test_the_forced_case_sizes_are_stated_by_name` | 218 | oracle_smoke |  | STAY-SINGLE | signature fact (keyword-only) |  | 0.00 |
| operators/test_burgers_forced_case.py | `test_every_registered_dataset_is_named_once` | 315 | oracle_smoke | Forced dataset is registered | STAY-SINGLE | registry fact |  | 0.00 |
| operators/test_burgers_forced_case.py | `test_every_forcing_to_solution_row_says_how_its_forcing_was_obtained` | 325 | oracle_smoke | The derived and drawn forcings are distinguishable | STAY-SINGLE | registry fact |  | 0.00 |
| operators/test_deeponet.py | `test_the_deeponet_value_types_are_immutable` | 347 | oracle_smoke |  | FUZZ | one enumerated immutability test over the closed set of value types (parametrize), merging both modules' lists |  | 0.00 |
| operators/test_operators.py | `test_operator_dataset_shapes` | 185 | oracle_smoke | Targets are evaluated, not pre-sampled | FOLD-MACHINE | SensorsAndDatasets: shapes as invariant of build_a_dataset |  | 0.00 |
| operators/test_operators.py | `test_spectralsvr_operator_learns_antiderivative` | 198 | oracle_analytic | Predict after fit;Query points are not restricted to the training grid | FUZZ | fuzz seeds/sizes; current seed pinned | CLOSED_FORM_TOLERANCE=0.70 is measured over seeds 0-9: pin verbatim, remeasure over the fuzzed range | 0.02 |
| operators/test_operators.py | `test_an_unfitted_operator_cannot_predict` | 243 | oracle_smoke | Predicting before fitting cannot be stated;Fitting yields the thing that predicts | STAY-SINGLE | type-level protocol fact on one adapter; could enumerate both adapters |  | 0.00 |
| operators/test_operators.py | `test_benchmark_returns_metric_table` | 259 | oracle_smoke | One row per operator, same metrics | FOLD-MACHINE | BenchmarkRuns: rules add SpectralSVR/Constant/Exploding operator; invariants one row per operator, shared targets, an exploding fit propagates |  | 0.01 |
| operators/test_operators.py | `test_benchmark_propagates_a_failing_operator` | 325 | oracle_smoke |  | FOLD-MACHINE | BenchmarkRuns: rules add SpectralSVR/Constant/Exploding operator; invariants one row per operator, shared targets, an exploding fit propagates |  | 0.00 |
| operators/test_operators.py | `test_benchmark_scores_two_operators_against_the_same_targets` | 335 | oracle_consistency | Shared inputs and targets;Each method takes the view it needs | FOLD-MACHINE | BenchmarkRuns: rules add SpectralSVR/Constant/Exploding operator; invariants one row per operator, shared targets, an exploding fit propagates |  | 0.00 |
| operators/test_operators.py | `test_the_benchmark_script_states_what_its_table_establishes` | 366 | oracle_smoke | Scope of the claim is stated;Fit times are qualified;A single-seed table is not presented as a result;Run-to-run spread is stated per operator | STAY-SINGLE | docstring text fact |  | 0.00 |
| operators/test_operators.py | `test_benchmark_documents_its_measurement_protocol` | 386 | oracle_smoke | Fit time excludes prediction;A failing operator is not silently omitted from a run | STAY-SINGLE | docstring text fact |  | 0.00 |
| operators/test_operators.py | `test_a_missing_operator_extra_refuses_by_default` | 446 | oracle_smoke | A short table requires an explicit opt-in | FUZZ | enumerate the closed set (requested operators x extra present x allow-missing) by parametrize |  | 0.00 |
| operators/test_operators.py | `test_fit_seconds_measures_the_fit_and_not_the_prediction` | 502 | oracle_analytic | Fit time excludes prediction | STAY-SINGLE | wall-clock timing with sleeps; fuzzing multiplies sleep cost and timer noise | 0.25 s pauses: pin | 0.50 |
| operators/test_operators.py | `test_the_value_types_are_immutable` | 519 | oracle_smoke |  | FUZZ | one enumerated immutability test over the closed set of value types (parametrize), merging both modules' lists |  | 0.00 |
| operators/test_time_varying_scoring.py | `test_the_prediction_bound_sits_within_ten_times_the_recorded_worst` | 320 | oracle_consistency |  | STAY-SINGLE | recorded-measurement fact |  | 0.00 |
| problems/test_problems.py | `test_axis_diff_rejects_bad_order` | 89 | oracle_smoke |  | MERGE-EXISTING | test_axis_diff_matches_analytic: add refused orders as a fuzzed refusal over integers outside {1, 2}, 0 and 3 pinned |  | 0.00 |
| problems/test_problems.py | `test_antiderivative_rejects_bad_sizes` | 296 | oracle_smoke |  | FUZZ | fuzz the refused sizes / one-point domains; current literals pinned |  | 0.00 |
| problems/test_problems.py | `test_burgers_numerical_solution_dissipates_energy` | 411 | oracle_invariant |  | FUZZ | fuzz seeds/nu/modes in the unforced regime; current literal pinned | 1e-6 slack literal: measure | 0.03 |
| problems/test_problems.py | `test_manufactured_branch_rejects_a_solver` | 455 | oracle_smoke |  | FUZZ | enumerate the closed set (branch x solver given / omitted) by parametrize |  | 0.00 |
| problems/test_problems.py | `test_numerical_branch_uses_the_supplied_solver` | 471 | oracle_smoke |  | FUZZ | enumerate the closed set (branch x solver given / omitted) by parametrize |  | 0.02 |
| problems/test_problems.py | `test_numerical_burgers_agrees_with_an_independent_integrator` | 654 | oracle_independent |  | FUZZ | fuzz seeds inside the non-stiff regime the test states; current literal pinned | 1e-4 agreement and regime boundary are measured: pin verbatim | 0.29 |
| problems/test_problems.py | `test_manufactured_pair_satisfies_the_continuous_equation` | 749 | oracle_analytic |  | FUZZ | fuzz seeds/bands; current case pinned | MANUFACTURED_RESIDUAL_TOLERANCE measured: pin verbatim | 0.09 |
| problems/test_problems.py | `test_continuous_residual_detects_a_corrupted_forcing` | 775 | oracle_analytic |  | FUZZ | fuzz seeds/bands; current case pinned | MANUFACTURED_RESIDUAL_TOLERANCE measured: pin verbatim | 0.02 |
| problems/test_problems.py | `test_generate_rejects_a_time_domain_without_a_step` | 1131 | oracle_smoke |  | FUZZ | fuzz the refused sizes / one-point domains; current literals pinned |  | 0.00 |
| problems/test_problems.py | `test_numerical_burgers_reproduces_manufactured_solution` | 1416 | oracle_analytic | Round-trip matches the manufactured solution;Exact solution reached within time-stepping error | STAY-SINGLE | convergence studies over a fixed step sweep with recorded constants (travel, orders, standing-still error); fuzzing changes the measured values | measured constants MANUFACTURED_ROUND_TRIP_*, FROZEN_FORCING_ORDERS, STANDING_STILL_WORST_ERROR: pin verbatim | 0.29 |
| problems/test_problems.py | `test_round_trip_is_not_satisfied_by_a_solver_that_stands_still` | 1459 | oracle_invariant | The round-trip comparison is not satisfied by standing still;A full-period comparison states what the solver had to do | STAY-SINGLE | convergence studies over a fixed step sweep with recorded constants (travel, orders, standing-still error); fuzzing changes the measured values | measured constants MANUFACTURED_ROUND_TRIP_*, FROZEN_FORCING_ORDERS, STANDING_STILL_WORST_ERROR: pin verbatim | 0.37 |
| problems/test_problems.py | `test_numerical_burgers_round_trip_converges_at_the_integrator_order` | 1477 | oracle_invariant | Round-trip converges at the integrator's order;Rate, not just size, is asserted | STAY-SINGLE | convergence studies over a fixed step sweep with recorded constants (travel, orders, standing-still error); fuzzing changes the measured values | measured constants MANUFACTURED_ROUND_TRIP_*, FROZEN_FORCING_ORDERS, STANDING_STILL_WORST_ERROR: pin verbatim | 0.59 |
| problems/test_problems.py | `test_numerical_burgers_forcing_frozen_at_step_start_loses_the_order` | 1499 | oracle_invariant | Forcing is evaluated at stage times, not step starts | STAY-SINGLE | convergence studies over a fixed step sweep with recorded constants (travel, orders, standing-still error); fuzzing changes the measured values | measured constants MANUFACTURED_ROUND_TRIP_*, FROZEN_FORCING_ORDERS, STANDING_STILL_WORST_ERROR: pin verbatim | 0.59 |
| problems/test_problems.py | `test_generation_survives_a_process_default_pointing_elsewhere` | 1568 | oracle_smoke | A workload placed against the process default still runs | FUZZ | enumerate DEVICES_PRESENT x process default (closed set) by parametrize |  | 0.00 |
| test_oracle_coverage.py | `test_every_module_has_an_anchor` | 55 | oracle_invariant |  | STAY-SINGLE | repository guard |  | 0.05 |
| test_oracle_coverage.py | `test_exempt_list_is_current` | 73 | oracle_invariant |  | STAY-SINGLE | repository guard |  | 0.00 |
| test_oracle_coverage.py | `test_mms_markers_are_retired` | 87 | oracle_invariant |  | STAY-SINGLE | repository guard |  | 0.05 |
| test_oracle_coverage.py | `test_oracle_marker_names_match_conftest` | 98 | oracle_invariant |  | STAY-SINGLE | repository guard |  | 0.05 |
| test_oracle_coverage.py | `test_setup_enforcement_rejects_unmarked_test` | 125 | oracle_smoke |  | FUZZ | one pytester test enumerating marker sets (none, one each, two) by parametrize |  | 0.00 |
| test_oracle_coverage.py | `test_setup_enforcement_rejects_multiple_oracle_markers` | 134 | oracle_smoke |  | FUZZ | one pytester test enumerating marker sets (none, one each, two) by parametrize |  | 0.00 |
| test_oracle_coverage.py | `test_setup_enforcement_accepts_single_oracle_marker` | 143 | oracle_smoke |  | FUZZ | one pytester test enumerating marker sets (none, one each, two) by parametrize |  | 0.00 |
| test_prose_budget.py | `test_no_file_carries_more_prose_than_its_budget` | 88 | oracle_consistency |  | STAY-SINGLE | repository guard |  | 0.15 |
| test_prose_budget.py | `test_no_budget_holds_slack` | 98 | oracle_consistency |  | STAY-SINGLE | repository guard |  | 0.16 |
| test_prose_budget.py | `test_every_ruff_docstring_finding_lands_in_a_counted_docstring` | 108 | oracle_independent |  | STAY-SINGLE | repository guard |  | 0.34 |
| test_prose_budget.py | `test_a_cyclic_reexport_between_packages_terminates` | 601 | oracle_smoke |  | STAY-SINGLE | one fixed malformed package; Python cannot import it, so the fuzzed export test cannot host it |  | 0.00 |
| test_scenario_coverage.py | `test_scenario_names_are_unique` | 205 | oracle_invariant |  | STAY-SINGLE | repository guard |  | 0.00 |
| test_scenario_coverage.py | `test_every_stated_scenario_is_covered_or_exempt` | 215 | oracle_consistency |  | STAY-SINGLE | repository guard |  | 0.05 |
| test_scenario_coverage.py | `test_no_exemption_is_stale` | 227 | oracle_consistency |  | STAY-SINGLE | repository guard |  | 0.05 |
| test_scenario_coverage.py | `test_every_in_flight_exemption_names_a_scenario_still_in_flight` | 242 | oracle_consistency |  | STAY-SINGLE | repository guard |  | 0.00 |
| utils/test_band.py | `test_the_forms_that_promise_a_real_result_are_exactly_two` | 76 | oracle_invariant |  | STAY-SINGLE | type/export fact |  | 0.00 |
| utils/test_band.py | `test_the_raw_form_states_the_guarantee_it_does_not_make` | 214 | oracle_smoke |  | STAY-SINGLE | docstring text fact |  | 0.00 |
| utils/test_band.py | `test_exported_from_the_package_root` | 225 | oracle_smoke |  | STAY-SINGLE | type/export fact |  | 0.00 |
| utils/test_metric_anchors.py | `test_get_metrics_hand_computed_values` | 20 | (module mark) | Neither replaces the other | FUZZ | fuzz arrays against independent numpy formulas; the hand literal kept as an analytic @example | ORACLE RISK: analytic only at the literal; keep it pinned | 0.00 |
| utils/test_metric_anchors.py | `test_get_metrics_rrse_near_constant_column` | 51 | (module mark) |  | FUZZ | fuzz near-constant columns; the regression literal pinned | regression case: pin verbatim | 0.00 |
| utils/test_metric_anchors.py | `test_get_metrics_rse_consistent_with_definition_rrse_squared` | 68 | (module mark) |  | MERGE-EXISTING | test_get_metrics_default_keys: add rrse**2 == rse and r2 NaN at one row as assertions/@example |  | 0.00 |
| utils/test_metric_anchors.py | `test_get_metrics_nan_count` | 75 | (module mark) |  | FUZZ | fuzz NaN positions / injected metric names; literals pinned |  | 0.00 |
| utils/test_solvers.py | `test_etdrk4_order_with_time_dependent_forcing` | 175 | oracle_invariant |  | STAY-SINGLE | convergence sweep with a recorded order bound | order > 3.9: pin verbatim | 0.13 |
| utils/test_solvers.py | `test_etdrk4_complex_weights_match_closed_form` | 194 | oracle_analytic |  | FUZZ | fuzz eigenvalues/step sizes; current case pinned | 1e-10 closed-form agreement: measure over range | 0.00 |
| utils/test_solvers.py | `test_etdrk4_real_weights_unchanged` | 206 | oracle_consistency |  | FUZZ | fuzz eigenvalues/step sizes; current case pinned | 1e-10 closed-form agreement: measure over range | 0.00 |
| utils/test_solvers.py | `test_etdrk4_rejects_non_uniform_grid` | 232 | oracle_smoke |  | FUZZ | one solver refusal table, enumerated kinds, fuzzed triggering inputs |  | 0.00 |
| utils/test_solvers.py | `test_etdrk4_requires_matching_shapes` | 245 | oracle_smoke |  | FUZZ | one solver refusal table, enumerated kinds, fuzzed triggering inputs |  | 0.00 |
| utils/test_solvers.py | `test_conversion_integrates_complex_spectral_state` | 281 | oracle_smoke |  | MERGE-EXISTING | test_conversion_equals_the_combined_right_hand_side: add complex state as @example |  | 0.08 |
| utils/test_solvers.py | `test_conversion_forwards_the_time_grid_unchanged` | 299 | oracle_smoke |  | FUZZ | fuzz grids; identity of forwarded objects asserted per draw |  | 0.00 |
| utils/test_utils.py | `test_accelerator_names_whichever_one_is_present` | 115 | oracle_smoke | Naming the accelerator is a statement, not a search | FOLD-MACHINE | DeviceStatements (new): rules state host/index/unstated, repoint default, make a generator; invariants resolve/name/refuse/agree per statement |  | 0.00 |
| utils/test_utils.py | `test_a_stated_host_survives_an_available_accelerator` | 128 | oracle_smoke | The host is not overridden | FOLD-MACHINE | DeviceStatements (new): rules state host/index/unstated, repoint default, make a generator; invariants resolve/name/refuse/agree per statement |  | 0.00 |
| utils/test_utils.py | `test_two_stated_indices_that_differ_are_a_disagreement` | 146 | oracle_smoke | A stated index is part of the statement | FOLD-MACHINE | DeviceStatements (new): rules state host/index/unstated, repoint default, make a generator; invariants resolve/name/refuse/agree per statement |  | 0.00 |
| utils/test_utils.py | `test_an_unstated_index_agrees_with_a_stated_one` | 156 | oracle_smoke | A stated index is part of the statement | FOLD-MACHINE | DeviceStatements (new): rules state host/index/unstated, repoint default, make a generator; invariants resolve/name/refuse/agree per statement |  | 0.00 |
| utils/test_utils.py | `test_a_random_source_device_is_invisible_to_a_type_checker` | 183 | oracle_smoke | A mismatched random source names both devices | FOLD-MACHINE | DeviceStatements (new): rules state host/index/unstated, repoint default, make a generator; invariants resolve/name/refuse/agree per statement |  | 0.00 |
| utils/test_utils.py | `test_get_metrics_single_sample_r2_is_nan` | 218 | oracle_smoke |  | MERGE-EXISTING | test_get_metrics_default_keys: add rrse**2 == rse and r2 NaN at one row as assertions/@example |  | 0.00 |
| utils/test_utils.py | `test_get_metrics_injected_metric_set` | 225 | oracle_smoke |  | FUZZ | fuzz NaN positions / injected metric names; literals pinned |  | 0.00 |
| utils/test_utils.py | `test_standard_scaler_tuple_and_subset` | 340 | oracle_consistency |  | FOLD-MACHINE | ScalerLifecycle (new): rules fit, transform tuple/subset, save/load, misuse; invariants inverse(transform) identity, loaded equals original, misuse refused |  | 0.00 |
| utils/test_utils.py | `test_standard_scaler_consistency_checks` | 352 | oracle_smoke |  | FOLD-MACHINE | ScalerLifecycle (new): rules fit, transform tuple/subset, save/load, misuse; invariants inverse(transform) identity, loaded equals original, misuse refused |  | 0.00 |
| utils/test_utils.py | `test_standard_scaler_save_load` | 362 | oracle_consistency |  | FOLD-MACHINE | ScalerLifecycle (new): rules fit, transform tuple/subset, save/load, misuse; invariants inverse(transform) identity, loaded equals original, misuse refused |  | 0.00 |
| utils/test_utils.py | `test_to_mag_angle_idempotent_on_real` | 398 | oracle_smoke |  | FUZZ | fuzz arrays over the existing real_arrays/complex strategies; current literals pinned |  | 0.00 |
| utils/test_utils.py | `test_to_mag_angle_handles_any_number_of_mode_axes` | 408 | oracle_analytic |  | MERGE-EXISTING | test_every_conversion_* (polar is in the registry): duplicates; remove after those are fuzzed |  | 0.00 |
| utils/test_utils.py | `test_from_mag_angle_inverts_to_mag_angle` | 420 | oracle_analytic |  | MERGE-EXISTING | test_every_conversion_* (polar is in the registry): duplicates; remove after those are fuzzed |  | 0.00 |
| utils/test_utils.py | `test_polar_encoding_precision_corresponds` | 429 | oracle_invariant |  | MERGE-EXISTING | test_every_conversion_* (polar is in the registry): duplicates; remove after those are fuzzed |  | 0.00 |
| utils/test_utils.py | `test_polar_encoding_preserves_device` | 442 | oracle_invariant |  | MERGE-EXISTING | test_every_conversion_* (polar is in the registry): duplicates; remove after those are fuzzed |  | 0.00 |
| utils/test_utils.py | `test_from_mag_angle_idempotent_on_complex` | 450 | oracle_smoke |  | FUZZ | fuzz arrays over the existing real_arrays/complex strategies; current literals pinned |  | 0.00 |
| utils/test_utils.py | `test_euler_solver_requires_multiple_points` | 475 | oracle_smoke |  | FUZZ | one solver refusal table, enumerated kinds, fuzzed triggering inputs |  | 0.00 |
| utils/test_utils.py | `test_zero_coeff_per_axis_bands` | 526 | oracle_analytic |  | FOLD-MACHINE | BandLimiting: per-axis bands as a rule, bare count as a refusal pin |  | 0.00 |
| utils/test_utils.py | `test_zero_coeff_rejects_a_bare_count` | 537 | oracle_smoke |  | FOLD-MACHINE | BandLimiting: per-axis bands as a rule, bare count as a refusal pin |  | 0.00 |
| utils/test_utils.py | `test_conversion_registry_covers_every_exported_conversion` | 600 | oracle_invariant |  | STAY-SINGLE | registry fact |  | 0.00 |
| utils/test_utils.py | `test_every_conversion_accepts_any_number_of_mode_axes` | 618 | oracle_invariant |  | FUZZ | replace internal loops with @given over shapes x dtypes, registry entries enumerated by parametrize |  | 0.00 |
| utils/test_utils.py | `test_every_conversion_preserves_precision` | 630 | oracle_invariant |  | FUZZ | replace internal loops with @given over shapes x dtypes, registry entries enumerated by parametrize |  | 0.00 |
| utils/test_utils.py | `test_every_conversion_preserves_device` | 644 | oracle_invariant |  | FUZZ | replace internal loops with @given over shapes x dtypes, registry entries enumerated by parametrize |  | 0.00 |
