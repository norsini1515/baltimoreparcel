# Urban Economics Final Project — Brainstorming Session Summary

## Project Overview

**Course:** Urban Economics
**Assignment:** Final Project — Urban Policy Analysis
**Student Location:** Baltimore, Maryland

### Assignment Requirements
- 5,000–6,500 words
- At least two original figures or tables
- At least three academic citations
- Must reference at least one urban economic model from the course
- Choose between **Option 1** (propose a policy) or **Option 2** (evaluate an existing policy)
- Proposal due first: 400–700 words covering city, policy question, models, and data sources

---

## Decision Log

### City: Baltimore, MD ✅
Chosen for several reasons:
- Student lives there — firsthand familiarity
- Potential to share with Baltimore City Planning Department or BDC
- Analytically rich: dramatic neighborhood inequality, gentrification pressure, large vacant property stock, active policy interventions
- Strong data availability via Open Baltimore and SDAT

### Option: Option 2 — Evaluate an Existing Policy ✅
Initially considered Option 1 (propose a policy) for flexibility, but Option 2 was favored because:
- Existing policies provide observable outcomes to anchor figures
- More concrete empirical grounding
- A hybrid framing is possible: evaluate an existing policy and propose an extension or reform

### Policy Areas Explored
Three policies were considered before a final decision:

1. **Vacants to Value (V2V)** — seriously considered
2. **Enterprise Zone Designations (EZD)** — **CHOSEN** ✅
3. **INSPIRE School Planning Corridors** — dropped early

---

## Policy Exploration: Vacants to Value (V2V)

### What it is
Baltimore's V2V program launched in 2010 under Mayor Rawlings-Blake. It uses code enforcement, receiverships, city-owned property disposition, and homebuyer incentives to reduce vacant properties and return them to productive use.

### Why it was considered
- Strong data availability: Open Baltimore Vacant Building Notices (VBN) dataset
- Parcel-level join possible via `BLOCKLOT` field
- Student's existing SDAT parcel panel (2003–2024) is directly applicable
- BNIA-JFI conducted a formal quantitative evaluation (2009–2015) providing academic scaffolding

### VBN Dataset Schema (Open Baltimore)
Confirmed via ArcGIS REST API:
- `NoticeNum` — unique notice identifier
- `DateNotice` — when vacant building notice was issued
- `DateCancel` — when notice was cancelled (property resolved)
- `DateAbate` — when notice was abated
- `BLOCKLOT` — parcel identifier (direct join key to SDAT data)
- `Neighborhood` — neighborhood name
- `HousingMarketTypology2023` — market strength classification
- `NT` — notice type
- `Council_District`, `Address`

**Key capability:** For each year in the parcel panel, a parcel can be classified as "vacant" if `DateNotice ≤ year` AND (`DateCancel` or `DateAbate` is null OR `> year`). This reconstructs a yearly vacancy boolean per parcel.

### Why V2V was ultimately not chosen
- Narrower audience (Baltimore-specific program)
- Less publication potential compared to EZ literature
- EZ offers richer natural experiment with multiple treatment waves
- V2V story is cleaner but less intellectually ambitious

### V2V Data Note
Vacancy data is still worth pulling. It serves a natural role as a **descriptive characteristic** of EZ neighborhoods in the background section — showing pre-treatment distress conditions. If vacancy trends diverge across EZ vs. non-EZ areas post-designation, it becomes a useful secondary outcome. Pull it, don't force it.

---

## Policy Exploration: Enterprise Zone Designations (EZD) ✅ CHOSEN

### What it is
Maryland's Enterprise Zone program provides **real property and state income tax credits** to businesses that create jobs and make capital investments in designated distressed areas. Baltimore City's EZ is administered by the Baltimore Development Corporation (BDC).

### Key Benefits
**Standard EZ:**
- 10-year real property tax credit on incremental increase from improvements
- 80% credit years 1–5, declining 10% annually years 6–10
- $1,000 income tax credit per new employee; $6,000 over 3 years for economically disadvantaged employees

**Focus Area (FA) — enhanced benefits:**
- 80% real property tax credit for full 10 years (no taper)
- Personal property tax credit (80% for 10 years on new investment)
- $1,500 income tax credit per new employee; $9,000 for economically disadvantaged employees

### Designation History (Baltimore City)

| Wave | Year | Acres | Notes |
|------|------|-------|-------|
| Original | 2002 | Unknown | First designation; boundary documentation not found online |
| Update | 2012 | 13,453 | Last pre-2022 update |
| Redesignation | 2022 | ~16,760 | +3,271 acres; significant boundary changes |

**2022 Areas Added:**
Belair Road, Downtown, Edmondson Square, Falls Road, Fayette Street, Frankford Avenue, Greenmount Avenue, Harford Corridor, Highlandtown, Greektown, Port Covington, Reisterstown Road, Sinclair Lane, North Washington Street, York Road Corridor

**2022 Areas Removed:**
Harbor Point, parts of Canton, parts of Little Italy (removed because they'd grown too much — the EZ working as intended)

**2022 Focus Areas (confirmed via Maryland Department of Commerce, July 2022):**
Jones Falls, Oldtown, Carroll-Camden, Central West, Holabird-Orangeville, South Industrial (newly added)

### Why EZD was chosen
- **Multi-wave natural experiment:** Three designation events (2002, 2012, 2022) within the 2003–2024 panel
- **De-designation angle:** Areas removed in 2022 (Harbor Point, Federal Hill) provide a natural counterfactual — if they appreciated sharply post-2012 relative to areas that stayed in, that's a clean result
- **Richer treatment variation:** Parcels fall into different treatment histories (continuous since 2002, added in 2012, added in 2022, or lost designation in 2022)
- **Stronger publication potential:** Place-based policy literature is active in urban economics (Busso, Gregory & Kline 2013; Neumark & Kolko 2010; Ham et al.)
- **Local relevance:** BDC administers the program for Baltimore City; results directly useful for the next redesignation cycle (~2032)
- **Maryland-level program with city-level administration** — doesn't reduce planning department relevance

---

## Data Infrastructure

### Student's Existing Parcel Panel (`baltimoreparcel` repo)
GitHub: [https://github.com/norsini1515/baltimoreparcel](https://github.com/norsini1515/baltimoreparcel)

**Key files:**
- `engineer_panel.py` — core analysis functions
- `gis_utils.py` — spatial utilities
- `config.py`, `directories.py` — configuration

**Key functions already built:**
- `log_value()` — log-transforms assessed value fields
- `calculate_change()` — vectorized long-form year-over-year log change calculator; handles both numeric and string fields; outputs per-year change rates
- `summarize_field()` — aggregates mean, median, std, net growth by grouping fields
- `spatial_join_with_neighborhoods()` — joins parcel data to Baltimore NSA boundaries
- `enrich_change_gdf()` — merges additional fields onto change panel
- `to_real_data()` — deflates nominal values using price index

**Panel coverage:** SDAT assessed values, 2003–2024, parcel level
**Value fields tracked:** Land value, improvement value, total value (all available)
**Parcel ID:** `ACCTID` and `BLOCKLOT`

### Outcome Variable Decision
**Total assessed value** chosen as primary outcome because:
- Captures both direct improvement effect (firms investing → triggering tax credit) and spillover land value appreciation
- Bigger signal than land value or improvement value alone
- Does not create endogeneity — selection bias is a causal identification issue, not a measurement problem, and the assignment does not require causal identification

### Real vs. Nominal Values
Student previously deflated SDAT values using **BLS Metropolitan price data** (base year ~2017) for the Geospatial Statistics course project. Key finding: nominal values increased over the panel period, but **real values declined for Baltimore broadly**.

**Implications for EZ paper:**
- Deflating is the more honest analysis — nominal appreciation that merely keeps pace with inflation doesn't indicate policy success
- Sharpens the research question: "Did EZ parcels appreciate *relative to the metro price trend*?" — closer to a treatment effect framing
- Dampening concern cuts both ways: if real values declined citywide but EZ parcels declined *less*, that's still a positive finding
- Base year is easy to adjust since the full BLS index series is already in hand — no chaining needed
- Possible limitation: metro-level deflator may not capture neighborhood-specific price dynamics; worth flagging transparently

---

## Analytical Framework

### Proposed EZ Treatment Variable Construction
Rather than relying on BNIA cohort classifications, build the treatment variable from scratch using:
1. Spatial join of EZ polygon vintages → NSA boundaries → lookup table per wave
2. Stack into a boolean timeseries per neighborhood: `EZ_2002`, `EZ_2012`, `EZ_2022`
3. Derive `EZ_ACTIVE` for each parcel-year based on applicable wave
4. Merge onto parcel panel on `NEIGHBORHOOD`

**Resulting schema:**
```
ACCTID | NEIGHBORHOOD | YEAR | LOG_NFMTTLVL | EZ_2002 | EZ_2012 | EZ_2022 | EZ_ACTIVE
```

### Potential Three-Tier Analysis (if FA boundary data found)
- Focus Area parcels (highest treatment intensity)
- EZ-only parcels (standard treatment)
- Non-EZ parcels (control)

If total assessed value responds to treatment intensity in a gradient, that's a clean descriptive story mapping onto the place-based development model.

### Models to Reference
- **Place-based economic development policies** (primary)
- **Agglomeration economies** (secondary — EZ theory predicts firm clustering)
- **Rosen-Roback spatial equilibrium** (if framing as amenity/wage shock to neighborhood)
- **Housing supply and land constraints** (supporting)

### Selection Bias
EZ designation is not random — zones are chosen because they're distressed, so pre-treatment values in treated areas are expected to be lower. This is a threat to causal identification, not a measurement problem. Strategy:
- Acknowledge selection bias explicitly in the Discussion section
- Discuss how DiD on the 2022 boundary changes or using removed areas as counterfactuals would address it in a full causal study
- This shows econometric sophistication without requiring execution of a full causal design (which the assignment does not require)

---

## Data Sources to Gather

| Dataset | Source | Status | Join Key |
|---------|--------|--------|----------|
| SDAT parcel panel (2003–2024) | Student's existing repo | ✅ Built | `ACCTID` / `BLOCKLOT` |
| EZ boundary — current (2022) | Open Baltimore | ✅ Available | Spatial |
| EZ boundary — 2012 vintage | Maryland Commerce / City Planning | ⚠️ Need to find | Spatial |
| EZ boundary — 2002 vintage | Maryland Commerce archives | ❌ Not found online | Spatial |
| EZ Focus Area boundaries | Maryland Commerce ArcGIS | ⚠️ Not in current Open Baltimore polygon | Spatial |
| NSA boundaries | Student's existing repo | ✅ Built | Spatial |
| Vacant Building Notices | Open Baltimore | ✅ Available | `BLOCKLOT` |
| BLS Metro price index | BLS | ✅ Student has from prior project | Time |

### Notes on 2002 Boundary
No archival shapefile or neighborhood-level list found online. Possible paths:
- Maryland Commerce records request
- Baltimore City Planning Department CityView GIS tool (lists EZ as a layer — may have historical vintages)
- DLS Library Maryland annual EZ status report PDFs

---

## Paper Structure Sketch

### Background (~20%)
- Baltimore context: population loss, industrial decline, distressed neighborhoods
- Why EZ designation? Selection criteria, distressed area characteristics
- Descriptive table: pre-treatment conditions by EZ status (median total value, vacancy rate, neighborhood typology)
- Show nominal vs. real value contrast to motivate the policy environment

### Policy Analysis (~60%)
- Explain EZ mechanism and three designation waves
- Theoretical predictions from place-based development model
- Figures: real log total value trends by EZ treatment group over time
- Connect empirical patterns to theoretical mechanisms
- Explore de-designation areas (Harbor Point, Federal Hill) as supporting evidence

### Discussion and Conclusion (~20%)
- What the model suggests about EZ effects
- Selection bias as a limitation — how causal identification would work ideally
- Trade-offs: do EZ benefits accrue to property owners or businesses?
- De-designation as policy success indicator
- Implications for Baltimore's next EZ cycle (~2032)
- Potential recommendation for BDC

---

## Key Academic Literature to Engage
- Busso, Gregory & Kline (2013) — federal Empowerment Zones, positive employment effects
- Neumark & Kolko (2010) — skeptical of California EZs, firms relocate rather than generate new activity
- Ham et al. — state-level EZ effects
- BNIA-JFI V2V Evaluation (2015) — useful for vacancy descriptive data
- Abell Foundation V2V Report — useful for Baltimore context

---

## Outstanding Questions / Next Steps
- [ ] Write proposal (400–700 words) covering city, question, models, data
- [ ] Find 2002 and 2012 EZ boundary data (CityView, Maryland Commerce, or records request)
- [ ] Check Maryland Commerce ArcGIS map for FA boundary layer as separate feature service
- [ ] Pull BLS metro price index back up and confirm base year
- [ ] Pull Vacant Building Notices dataset for descriptive use
- [ ] Confirm NSA names map cleanly to EZ documentation area names