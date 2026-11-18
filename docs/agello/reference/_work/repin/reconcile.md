# Re-pin of 2026-11-17: the reconciliation round

The first round ([brief.md](brief.md)) re-verified the claims whose cited code changed, each
verifier on its own pages, and the merge is done. Two things are left, and your task file
(`_work/repin/r<n>.json`) lists both for your pages:

1. **Copies.** Many claims are tagged on several pages: a table in `facts.md`, `configuration.md`,
   `stack.md` or `troubleshooting.md` repeats a claim whose home is another page. When a verifier
   corrected a claim, it fixed the sentence on its own page, not the copies. Each entry under
   `copies` gives the page, the line, the claim, its verdict, and its text before and after the
   re-pin. Make each copy agree with the claim's text now: often the change is only wording (the old
   pin named, "Google's v0.18.0 binary" as what ships) and the copy may already be fine; sometimes
   the behavior changed and the copy must say the new thing. For a `wrong` claim, remove its tag and
   what the copy says on its strength. Keep the page's phrasing and format (a table row stays a
   table row); re-cite any link you change with `cite.py`.
2. **Claims the first round didn't list.** Their cited lines didn't change, but their meaning may
   have: they speak of what Agello ships, Google's binary, the old forks or what the app does, and
   all of that moved (Agello has shipped our runtime since 2026-10-28; the release it ships now is
   `agello-one-20261115`, with `main`'s Swift wrapper at `82b05ea1`, the same files as at the pin;
   Agello's app repository is `/Users/patricedery/Coding projects/iRT`, read only). Give each a
   verdict at the pin in the batch named in your task file, as the brief says (corrected_text in
   full, corrected_sources re-cited when any source changes), and fix its sentences on your pages.

Then, still on your pages: if you meet a sentence that the re-pin made stale and that no listed
claim covers, fix it; if it's a tagged claim, give it a verdict in your batch too.

Rules: the brief's "Wording at the new pin" and its "Pages" bullets hold. Edit only your pages and
your batch files; don't merge, commit, build, run the engine or post anything. Keep scratch in a
folder named after you inside the scratchpad, and delete only that folder.

Done when `claims.py check-verdicts <group> z-repin-<you> --ids ...` passes for each batch, and
`check_refs.py --only <your pages>` (without `--no-tags`: the ledger is merged) reports no errors
other than tags whose claims your own new verdicts change (the main loop merges again). Report:
copies changed (count, and the ones where behavior changed), verdicts per group, anything unsettled.
