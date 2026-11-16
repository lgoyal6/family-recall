# Recall Console

The interface for Family Recall: the incident, the model family, screening verdicts, the transfer matrix, the answers themselves,
the cost of each option, release receipts, and a chat that answers only from the measured results.

```
npm install
npm run dev            # http://localhost:3000
```

Data lives in `public/data/*.json` and is produced by `../recall/build_ui_data.py` from the study's outputs; nothing is typed in by hand.
"Ask the evidence" needs `GMI_API_KEY` in `.env.local` (or as a Space secret). The `Dockerfile` builds a Hugging Face Space (port 7860).

Built with Next.js, shadcn/ui and Tailwind on the Azure Pro theme, using 21st.dev components (orb avatar, Siri orb, agent state,
sidebar, chat empty state, loader, roadmap). Sibling colors pass a colorblind-safety palette check in light and dark mode.
