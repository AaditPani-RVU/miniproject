# Document corpus

Drop `.pdf`, `.txt` or `.md` files here (ICAR bulletins, state agriculture department guides, KVK
package-of-practices, PM-KISAN / soil health card scheme pages). Then restart the server or call
`POST /api/reindex`.

Tips
- Pick one region and 2-3 crops. A focused, high-quality corpus beats a big noisy one.
- Prefer English or Hindi text PDFs. Scanned PDFs need OCR first.
- Keep a list of every source (title, URL, date) for the report.

## Current sources
1. `iiwbr_wheat_cultivation_in_india_2014.pdf`: Kumar, A. et al., *Wheat Cultivation in India: Pocket Guide*, Extension Bulletin 52, ICAR-Directorate of Wheat Research, Karnal, Sept 2014 (via ResearchGate). English, 41 pages.
2. `iiwbr_cleaned_tables.md`: hand-cleaned restatement of the tables and 4W-1H chart from source 1 (Tables 1-6 and the practical chart), written because PDF extraction scrambles them. Table 5 was checked against the page image. No numbers were added.
3. `indian_farming_2023_wheat_fertilizer_practices_nwpz.pdf`: Tripathi, S.C., Kumar, N., Samota, S.R., "Farmer's fertilizer practices in wheat", *Indian Farming* 73(9): 37-39, Sept 2023 (epubs.icar.org.in). English, 3 pages. Karnal (Haryana) case study; only a regional supporting source for UP.

Known gaps: no Hindi sources, no other crops, and no state-specific (UP) package of practices.
