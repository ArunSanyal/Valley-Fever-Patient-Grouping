
# Valley Fever Patient Clustering

## Intro

This project is a small Python pipeline that groups synthetic Valley Fever patients into meaningful clusters, using both their symptoms and their social determinants of health, such as housing stability and insurance status.

No real patient data is used anywhere. All 200 patients are fully synthetic, generated from 4 hidden patient archetypes so the clustering method could be tested against a known structure before ever being considered for real data. The goal is to see whether patients group into distinct, real-world-meaningful groups that a clinician or outreach coordinator could actually act on, not just groups that look separated on paper.

## Files in this repo

| File                                 | What it does                                                       |
| ------------------------------------ | ------------------------------------------------------------------ |
| **generate_synthetic_data.py** | Creates 200 synthetic patients from 4 hidden archetypes            |
| **preprocess.py**              | Prepares the data for clustering (scaling, formatting)             |
| **cluster.py**                 | Runs the K-Prototypes clustering algorithm                         |
| **evaluate.py**                | Checks how good the clustering is, picks the best number of groups |
| **code.ipynb**                 | Runs the full pipeline with charts, this is the main demo          |
| **synthetic_patients.csv**     | Output of the data generator                                       |
| **clustered_patients.csv**     | Final output, each patient with their assigned group               |
| **README.md**                  | This file                                                          |

## How to run

1. Install requirements: **pip install pandas numpy scikit-learn scipy kmodes matplotlib**
2. Run each stage in order:
   - **python generate_synthetic_data.py**
   - **python preprocess.py**
   - **python cluster.py**
   - **python evaluate.py**
3. Open **code.ipynb** and run all cells to see the full walkthrough with charts.

## What output you get from each stage

**generate_synthetic_data.py**
Builds 200 fake patients around 4 hidden character types, called archetypes: a delayed-diagnosis vulnerable group, a stable well-managed group, an elderly moderate-risk group, and a young high-exposure group. Each patient is generated with some random variation so no two patients in the same archetype are identical.
Output: synthetic_patients.csv
What we got: 200 patients, roughly 45 to 55 per archetype, with realistic-looking symptom and social data.

**preprocess.py**
Gets the data ready for clustering. Removes columns that should not be used for grouping, such as patient ID and the hidden archetype answer key, and scales the number-based columns so no single one, like distance to clinic in miles, automatically outweighs a smaller-scale column like symptom severity.
Output: used internally, not saved as its own file.

**cluster.py**
Runs K-Prototypes, a clustering method built to handle a mix of number columns and category columns at the same time. This step also saves an early, first-draft version of the results.
Output: an early version of clustered_patients.csv (later overwritten by evaluate.py with the final, corrected result).

**evaluate.py**
Checks the clustering three different ways: a statistical separation check (silhouette score), a check for how well the clustering rediscovered the 4 hidden archetypes (Adjusted Rand Index and Normalized Mutual Information), and a plain-language summary of what each group actually looks like.
Output: the final clustered_patients.csv, with each patient's real data, their group number, and how confident that assignment was.
What we got: the best number of groups turned out to be 6, not 4. Silhouette score peaked at 6 groups (about 0.297). ARI and NMI both came out around 0.43, meaning the clustering meaningfully rediscovered the hidden structure, though not a perfect match. 22 of the 200 patients (11%) were flagged as uncertain, meaning they sit close to the border between two groups instead of clearly belonging to one.

## Design choices and assumptions

**Why synthetic data:** real patient records are protected health information and would require IRB approval and formal data agreements to use, even for a class exercise. Since the point here is to practice clustering methods, not publish clinical findings, a fully synthetic dataset avoids that problem entirely. No row in this dataset corresponds to a real person.

**Why archetype-based generation:** an earlier version of the generator created every patient independently, with only small nudges between features. That produced a dataset with almost no real group structure, silhouette scores came out around 0.06, essentially no separation. Real patient populations usually do have some distinguishable subgroups, so this version instead builds each patient around one of 4 defined archetypes, with a hidden label kept aside so we could check afterward whether the clustering method actually finds something close to what was built in.

**Why K-Prototypes instead of plain K-Means:** the data mixes numbers, like symptom severity and distance to clinic, with categories, like housing status and insurance type. Plain K-Means only handles numbers. K-Prototypes is built specifically to measure similarity across both types at once, without needing to force categories into number form first.

**Why the distance formula needed fixing:** the first version of the evaluation script measured cluster quality using a distance calculation that did not exactly match what the clustering algorithm used internally. After checking the actual clustering library's source code, the calculation was corrected to match: squared distance for number columns, and category mismatches weighted by a factor called gamma, which the library calculates automatically. This is a precision fix, not a change to the clustering method itself.

**Why Adjusted Rand Index and Normalized Mutual Information:** since this dataset has a hidden answer key (the 4 archetypes), these two standard metrics give a numeric answer to "how well did the clustering rediscover the groups we built in," on top of just eyeballing a table. Both range from 0 (no better than random) to 1 (perfect match).

**Why per-patient uncertainty:** a single average score can look fine even when some individual patients are sitting right on the border between two groups. Checking each patient separately, instead of trusting one dataset-wide average, means borderline patients can be flagged honestly instead of treated as equally confident as clear-cut cases.

## Key finding

After fixing the distance calculation, the best number of groups changed from 4 to 6. This was not a mistake, it is a direct result of measuring correctly. Three of the six groups line up cleanly with three of the original four archetypes: one group is almost entirely the stable, well-managed archetype, one is mostly the young, high-exposure archetype, and one is mostly the elderly, moderate-risk archetype. The fourth archetype, the delayed-diagnosis vulnerable group, actually split into three separate groups once measured correctly, roughly meaning "long symptom duration and unemployed," "long distance from clinic and uninsured," and a smaller, more mixed group with the highest rate of prior misdiagnosis.

In plain terms, the corrected clustering found more specific structure than we originally built in on purpose, meaning delayed diagnosis and poor access to care are not one single pattern, but a few different ones, which could call for different kinds of outreach depending on which pattern a real patient matches.

## Note on AI assistance

Claude was used throughout this project to help think through design decisions, explain how parts of the code work, and suggest good approaches to structuring the pipeline. All final decisions on what to build, how to design it, and what tradeoffs to make were made by me.
