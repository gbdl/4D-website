# Publishing and local preview

This is a static website. GitHub Pages publishes the files directly from the root of `main`; no custom workflow or build command is needed.

## Publish

1. Commit and push the site to `main` on GitHub.
2. Open the repository's **Settings → Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**.
4. Select **main** and **/(root)**, then click **Save**.

The site will be available at https://gbdl.github.io/4D-website/. Subsequent pushes to `main` update it automatically. The empty `.nojekyll` file disables Jekyll processing so the static files, including the numerical companion's underscore-prefixed Python modules, are served as supplied. See [GitHub's publishing instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

Keep `.gitignore`: it excludes local environments, caches, common credential files and internal assessment notes from normal Git staging. Review `git add --dry-run .` before committing. All committed content is public in a public repository; there is no separate publication filter. The data and frozen weights are intentionally public.

If your commit email should remain private, set this repository's `user.email` to the `noreply` address provided in **GitHub → Settings → Emails** before committing. See [GitHub's commit email instructions](https://docs.github.com/en/account-and-profile/how-tos/email-preferences/setting-your-commit-email-address).

## Preview and check locally

From the repository root:

```sh
python3 tools/check_site.py
node tools/check_frontend.mjs
python3 -m http.server 8787 --bind 127.0.0.1
```

Open **http://127.0.0.1:8787/**. The server lets the interactive views fetch the bundled JSON files. Stop it with Ctrl-C. These checks run locally; branch publication does not run the custom checks automatically.
