# Investment Strategy Matching Frontend

Next.js frontend for the target repository. The frontend calls the FastAPI backend in `../api/` and does not contain strategy-matching business logic.

## Local Development

Start the API from the repository root:

```bash
uvicorn api.main:app --reload --port 8000
```

Start the frontend from this directory:

```bash
npm install
npm run dev
```

Open `http://localhost:3000`. `next.config.ts` rewrites `/api/*` to `http://localhost:8000/*`.

## Runtime Layout

- `app/`: Next.js App Router page and global styles.
- `components/`: dashboard, recommendation, chart, and control components.
- `lib/api.ts`: typed API client for the FastAPI backend.
