# Editorial review

## Assessment before revision

The manuscript has a coherent central distinction: sharing may have different value at a new monitoring location and at a previously observed location. The matched 2024 outcomes and history-aware baseline support this distinction. The main obstacles to reading were:

- **Repetition:** the contribution, the distinction between scenarios, and cautions about routes, forecasting and comparisons between modes recur in the introduction, literature review and discussion. Some repetition orients the reader; repeated restatements slow the argument.
- **Information order:** regional balance results precede the definitions of the regions and aggregation rules. Data conventions and descriptive findings occupy one long opening subsection. Model definitions, historical inputs and the scaling reference are also crowded together.
- **Terminology:** “positions”, “sites”, “locations”, “imaginary sensors” and “virtual counters” alternate for closely related concepts. The progression of input sets is sometimes described through unexplained shorthand such as “matched ladder”.
- **Interpretive transitions:** a gain relative to one learner's baseline is not the same as the lowest error across learners; pooled deterioration can coexist with local improvement. These distinctions are present but can be stated more directly.
- **Methodological gaps:** prediction-period conventions, the six bicycle-location exclusions, sharing-coverage screening and interval construction need author clarification. The original Section III points to night conventions supposedly explained in Section IV, but that explanation is absent.

## Editorial changes applied

- Revised the abstract, introduction and conclusion around the same scenario-specific claims and retained all reported quantitative findings.
- Presented the research questions separately and shortened repeated statements of contribution and scope.
- Separated rental-cohort definitions from descriptive results, and moved regional definitions before regional findings.
- Separated input sets and historical profiles from model definitions and the reference estimator.
- Standardised scenario terminology around “new locations” and “known locations”, defined “chord”, and replaced unnecessary shorthand.
- Clarified within-learner versus across-learner comparisons, pooled versus local evaluation, and possible mechanisms versus demonstrated findings.
- Preserved citations, numerical results, methodological qualifications and limitations. Generated tables, figures, bibliography and analysis code were not edited. No experiments were rerun.

## Questions requiring author clarification

These questions remain unresolved; no scientific choices have been inserted into the manuscript.

1. **Prediction periods — Sections III-C and IV-B.** What are the exact day/night boundaries and date-assignment rules for each target? The accompanying README describes a bicycle service date starting at 06:00, a motor-vehicle calendar date, and day hours 06–23. Should Section IV explicitly state these rules, including the assignment of the last 2023 bicycle night and treatment of clock-change periods? The descriptive maps instead use continuous 20–06 nights; these must remain distinguishable from prediction targets.

2. **Bicycle-location exclusions — Section IV-A.** Which six locations are excluded, and for what prespecified reasons? Are these exclusions separate from, or consequences of, the minimum-date, consistency and complete-period checks? “Independently of prediction error” does not identify the selection rule.

3. **Sharing coverage — Sections III-A–C and IV-C.** What operational rule defines “usable bicycle-sharing coverage” and a period affected by the “near-total source timestamp defect”? Which dates or periods does it exclude, and is the same rule used for descriptive matching and prediction? Please also confirm the basis for describing the October 2023 anomaly as a timestamp defect rather than more generally as missing source coverage.

4. **Sensitivity intervals — Section IV-E and Figure 6.** What quantiles or nominal coverage define the reported paired-resampling intervals? Does the same construction apply to the Geometry-minus-Endpoints interval mentioned in Section V-B? The manuscript specifies 2,000 paired resamples and fixed predictions, but not the interval endpoints' construction.

5. **Contextual data — Section IV-C.** What are the provenance and temporal/spatial matching rules for temperature, precipitation and holidays? Can these be specified from the sources actually used, without adding a new data source or changing the experiment? The current manuscript lists these predictors but does not document their origin.
