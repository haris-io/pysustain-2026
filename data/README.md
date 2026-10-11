# CivicLens scheme catalogue (`data/schemes.json`)

This folder holds the small, hand-curated catalogue that CivicLens retrieves from and evaluates. Read this file before changing the data or relying on it.

**Status at a glance (checked 2026-10-11): 0 of 4 records are `verified`. All 4 are `needs_review`.**
That is deliberate. Every scheme below has at least one required eligibility criterion that the eight-field profile cannot represent, so a rules-only `potential_match` would overstate certainty. The engine returns `not_verified` for all of them, so no record can produce a positive match until a human promotes it (see "Promoting a record").

## Scope and intended student persona

- Persona: a student in, or from, Tamil Nadu who describes their situation in plain language and wants to know which education-support schemes are worth looking into and what to prepare.
- Scope: up to five education-support schemes, preferring evidence quality over record count. Four are included.
- Not in scope: a comprehensive benefit directory, school-level scholarships, loans, coaching schemes, or any scheme whose official source could not be read.

## Records

scheme_id
Authority
Official source read
Rules encoded
Status

`tn_first_graduate_tuition_concession_engineering`
Directorate of Technical Education, Tamil Nadu
https://dte.tn.gov.in/schemes-engineering-students
3
`needs_review`

`pm_usp_csss_college_university_students`
Dept. of Higher Education, Ministry of Education (via National Scholarship Portal)
https://scholarships.gov.in/public/schemeGuidelines/CSSS_GUIDLINES_07022024_updated.pdf
2
`needs_review`

`tn_moovalur_ramamirtham_pudhumai_penn`
Government of Tamil Nadu (DIPR press release; scheme portal)
https://kygcms.tn.gov.in/cms_migrated/document/press_release/pr290723_e_1513.pdf
0
`needs_review`

`tn_tamizh_pudhalvan`
Social Welfare and Women Empowerment Dept., Tamil Nadu
https://stationeryprinting.tn.gov.in/extraordinary/2024/230_Ex_II_1_2024.pdf
0
`needs_review`

All source checks were performed on **2026-10-11**, and that is the `verified_on` value in each record. It records when the source was read, not that the record is `verified`.

### 1. First Generation Graduate Tuition Fee Concession (engineering colleges)

- **Checked on the official DTE page:** the concession applies to first-generation graduates admitted in the first year or directly into the second year of engineering colleges in Tamil Nadu through Single Window Counselling; a "No Graduate" certificate from the competent Revenue authority must be submitted at counselling; the college applies the concession and DTE reimburses it; amounts vary by college category.
- **Encoded rules:** `first_generation_student` equals true; `course_type` in engineering synonyms; `state` equals Tamil Nadu.
- **Documents and steps:** the "No Graduate" certificate, and two steps (obtain it, then submit it at counselling) taken directly from the page. No `reuse_group`.
- **Why `needs_review`:** admission through Single Window Counselling in the first or direct second year cannot be represented. The DTE page carries no date. The TNEA information brochure for the current cycle could not be opened (see "Sources that could not be read"). Only engineering colleges are covered by the checked page; concessions for other professional courses were not checked. Concession amounts on the page may be out of date.
- **Interpretation note:** the `state` rule means the state where the college is located. The page does not state a residence requirement.

### 2. PM-USP Central Sector Scheme of Scholarship for College and University Students (CSSS)

- **Checked in the official guideline (read in full):** top-80th-percentile Class XII criterion, regular degree courses only (no distance mode, no diploma), AICTE or regulator-recognised institutions with a valid AISHE code, not availing any other merit scholarship, State scholarship, fee waiver or reimbursement scheme, income up to Rs 4.5 lakh per annum (certificate only for fresh applicants), renewal conditions (50% marks, 75% attendance), rates of scholarship, online application on NSP.
- **Encoded rules:** `annual_family_income_inr` less than or equal to 450000; `education_level` in undergraduate, postgraduate.
- **Documents and steps:** income certificate (fresh applicants only), savings bank account in the student's own name, and one step (apply on NSP). No `reuse_group`.
- **Why `needs_review`:** the Class XII percentile, the "no other scholarship or fee waiver" condition, regular mode and institution recognition cannot be represented. The two encoded rules are only a subset of the criteria, and the percentile is the core of this merit scheme.
- **Timing seen on NSP (2026-10-11):** the scheme listing shows the 2026-27 window as "for Renewal" only (student applications until 31-10-2026, institute verification until 15-11-2026, nodal verification until 30-11-2026). No fresh-application window for this scheme was visible on that page. Dates change; re-check NSP.

### 3. Moovalur Ramamirtham Ammaiyar Higher Education Assurance Scheme (Pudhumai Penn)

- **Checked:** Press Release No. 1513 (29.07.2023, DIPR) for eligibility, benefit, duration limits and registration through the college nodal officer; the scheme portal https://www.pudhumaipenn.tn.gov.in was opened and confirms the Rs 1,000 per month benefit but gives no further criteria.
- **Encoded rules:** none. The main criteria (girl student, schooling from Class 6 to 12 in Government schools as defined, first higher education course only, regular mode, recognised institution) are not representable, and a partial course-level rule would wrongly exclude some covered courses.
- **Unresolved:** later changes, such as news reports of an extension to government-aided schools, were not verified from an official source. The press release says the incentive is payable in addition to other scholarships; this conflicts with the CSSS guideline (see relationships).

### 4. Tamizh Pudhalvan Scheme

- **Checked:** the 18 July 2024 Gazette notification (G.O. (Ms) No. 42) names the beneficiaries (boy students who studied in Government Schools of both media or Government Aided Schools, Tamil medium only, and are pursuing graduation, diploma or ITI), the Rs 1,000 per month incentive, and the Aadhaar requirement with alternative identity documents.
- **Encoded rules:** none (gender and school history are not representable).
- **Not established from an official source:** the class range of schooling, regular-mode or recognition conditions, and the application procedure. The scheme portal blocked automated access, so no application URL or steps are recorded.

## Relationships

Four relationship records exist, all `unknown` with `needs_review`. None is presented as compatible, restricted, prerequisite or complementary.

- CSSS to the First Graduate concession, Pudhumai Penn and Tamizh Pudhalvan, and Pudhumai Penn to CSSS. The CSSS guideline (para 4(iv)) excludes students availing State-run scholarships or fee waivers, while the Pudhumai Penn press release says that scheme is payable in addition to other scholarships. No authoritative source names the schemes together, so the interaction is not established.
- An absent relationship means unknown, not compatible.

## Limitations from the prototype profile

The profile has exactly eight fields. These criteria appear in the sources but have no field, so they are documented here instead of being encoded or dropped silently:

- gender (Pudhumai Penn, Tamizh Pudhalvan)
- type, medium and years of schooling (Pudhumai Penn, Tamizh Pudhalvan)
- Class XII percentile standing (CSSS)
- other scholarships or fee waivers already availed (CSSS)
- regular versus distance mode and institution recognition (CSSS, Pudhumai Penn)
- admission route, entry year and the Revenue-authority certificate itself (First Graduate concession)
- fresh versus renewal applicant status (CSSS)

Adding fields is a team decision; this file does not change the schema.

## Value vocabulary assumed by the rules

Rule values are lowercase except proper names. The engine should compare strings after trimming and case-folding.

- `education_level`: `undergraduate`, `postgraduate`
- `course_type`: `engineering`, `b.e.`, `b.tech`, `b.e./b.tech`
- `state`: `Tamil Nadu`

The extraction step needs to normalise user wording to these values. Any other wording will make a condition unknown or not satisfied.

## Promoting a record to `verified`

All three steps are required:

1. Re-open the official source and confirm each encoded rule, benefit, document and step still matches the current text. Update `verified_on` to the real check date.
2. Confirm every required criterion is represented by an encoded rule, or add the missing profile field. If any required criterion stays unrepresented, keep `needs_review`.
3. Change `verification_status` to `verified`. A verified record must keep non-empty `eligibility_rules`, each with a `source_url`.

The First Graduate concession is the closest candidate. Its three rules already match the official page, and promotion hinges on the team deciding the unrepresented admission-route criterion is acceptable, or on confirming it against the current TNEA brochure.

## Rechecking and updating

- Open the authority's own page or guideline. Do not rely on news reports, aggregators or search snippets.
- Fix any URL, amount, threshold or date that has changed, and update `verified_on`.
- Keep `reuse_group` off documents unless you have confirmed that one preparation task satisfies both requirements (same issuer, validity and content). None is currently set.
- Do not add a relationship type other than `unknown` without recorded official evidence.
- After editing, check that the file is a valid UTF-8 JSON array, IDs are unique, fields and operators are from the contract, dependencies and relationship targets exist, and every non-empty URL was actually checked.

## Sources that could not be read

- https://www.tneaonline.org (TNEA, run by DTE): blocked automated access.
- https://www.tamilpudhalvan.tn.gov.in and https://penkalvi.tn.gov.in: disallowed automated access.
- https://www.education.gov.in CSSS page: returned no readable content. The NSP-hosted guideline was used instead.
- https://scholarships.gov.in/scholarshipEligibility: returned no readable content.

Scholarship aggregator sites, news reports and admissions-portal explainers were seen during research but are not cited in the data and were not used as proof of any rule.

## Considered but not included

PM YASASVI (OBC, EBC and DNT), the Post Matric Scholarship for SC/ST students, AICTE Pragati and Swanath scholarships, and similar schemes depend on social category, gender or other attributes the profile does not hold. Their guidelines were not researched in detail.

## What this catalogue is not

- It is a four-record prototype, not a comprehensive benefit directory. Absence of a scheme here says nothing about a student's entitlement.
- It makes no claim of legal or policy completeness.
- Nothing in it is a decision. The responsible government authorities make all final eligibility and award decisions. A "potential match" from CivicLens is preliminary screening only.