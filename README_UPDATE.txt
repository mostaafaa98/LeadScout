LeadScout / MABS v3
====================

Changes:
- Search multiple countries/cities in one job: one location per line.
- Results target is per location.
- Two locations are searched in parallel by default.
- Detail pages are processed concurrently (4 workers by default).
- Adds "Search location" to every lead and to Excel.
- Client-side result search, location filter, minimum-rating filter, phone-only filter.
- Sort by name, category, phone, rating, reviews, address, or location.
- Dark/Light mode toggle with localStorage.
- Existing /api/location compatibility is preserved.
- Excel and CSV exports remain available.

Default concurrency:
  LEADSCOUT_LOCATION_CONCURRENCY=2
  LEADSCOUT_DETAIL_CONCURRENCY=4

To update MABS:
1. Extract this package.
2. Double-click UPDATE_MABS.bat.
3. Start MABS normally with:
   cd /d "C:\Users\hp\OneDrive\Desktop\MaBs"
   venv\Scripts\activate
   python main.py
4. Open:
   http://127.0.0.1:8000

Backups are created automatically before replacement.
