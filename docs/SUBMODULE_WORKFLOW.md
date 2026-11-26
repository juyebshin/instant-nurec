# instant-nurec as a submodule of world-x

This repo is used as a git submodule of the shared
[AXE-KAIST/world-x](https://github.com/AXE-KAIST/world-x) project, and it is
still a standalone fork at the same time.

```
world-x/                      origin = git@github.com:AXE-KAIST/world-x.git
  instant-nurec/              this repo (submodule, branch pmt-finetune)
    remotes:
      juyeb  = git@github.com:juyebshin/instant-nurec.git   <- push here
      origin = git@github.com:NVIDIA/instant-nurec.git      <- upstream, fetch only
```

world-x does not contain instant-nurec's code. It only records one commit
(a "gitlink") from `juyebshin/instant-nurec`, as set in `world-x/.gitmodules`:

```ini
[submodule "instant-nurec"]
	path = instant-nurec
	url = https://github.com/juyebshin/instant-nurec.git
	branch = pmt-finetune
```

## Rules

1. **Push instant-nurec to juyeb before you push world-x.** world-x only stores
   a commit hash. If that commit is not on `juyebshin/instant-nurec` yet,
   teammates get `fatal: remote error: upload-pack: not our ref` when they run
   `git submodule update`.
2. **Never push to `origin` (NVIDIA).** It is only for pulling upstream updates.
3. **Work on a branch, not on a detached HEAD.** `git submodule update` leaves
   the submodule detached. Run `git checkout pmt-finetune` (or create a feature
   branch) before you commit, or the commits are easy to lose.
4. **Keep `pmt-finetune` tracking `juyeb/pmt-finetune`.** `git submodule sync`
   rewrites the URL of the remote the current branch tracks, and falls back to
   `origin`. If the branch has no upstream, the NVIDIA `origin` URL gets
   overwritten with the juyeb URL. Check with `git branch -vv`.
5. **Bump the pointer in world-x on purpose.** A changed `instant-nurec` entry in
   `git status` of world-x means the checked-out commit differs from the one
   recorded. Commit it only when world-x should move to that commit.
6. **Large files stay out of git.** `data/`, `checkpoints/` and `outputs/` are in
   `.gitignore`. Share them over the shared filesystem, not through git.

## Day-to-day

### Work on instant-nurec

```bash
cd world-x/instant-nurec
git checkout pmt-finetune            # skip if already on it
# ... edit ...
git add <files> && git commit -m "..."
git push                             # -> juyeb/pmt-finetune
```

### Record the new commit in world-x

```bash
cd world-x
git add instant-nurec
git commit -m "Bump instant-nurec"
git push                             # -> AXE-KAIST/world-x
```

### Pull upstream NVIDIA changes

```bash
cd world-x/instant-nurec
git fetch origin
git merge origin/main                # or: git rebase origin/main
git push                             # -> juyeb, then bump world-x as above
```

## For teammates

### Clone

```bash
git clone --recursive git@github.com:AXE-KAIST/world-x.git
```

If you already cloned without `--recursive`:

```bash
cd world-x
git submodule update --init --recursive
```

### Stay in sync

After every `git pull` of world-x, check out the commit that world-x records:

```bash
git pull
git submodule update --init --recursive
```

Or run this once per machine, and `git pull` does it automatically:

```bash
git config --global submodule.recurse true
```

### Get the latest `pmt-finetune` before world-x records it

```bash
cd world-x
git submodule update --remote instant-nurec
```

This leaves the pointer changed in world-x. Do not commit it unless you mean to
bump world-x (see rule 5).

### Contribute to instant-nurec

You need write access to `juyebshin/instant-nurec`. If you don't have it, fork
the repo, push a branch to your fork, and open a PR to `juyebshin/instant-nurec`.

```bash
cd world-x/instant-nurec
git checkout -b feat/my-change
git remote add me git@github.com:<you>/instant-nurec.git   # only without write access
git push -u me feat/my-change                              # or: git push -u juyeb feat/my-change
```

## Local paths

The working copy lives at `/home1/irteam/workspace/world-x/instant-nurec`. The
old path `/home1/irteam/workspace/instant-nurec` is a symlink to it, so these
still work:

- the hard-coded paths in `configs/training/pmt_*.yaml`
- the `inurec_*` scripts in `mapnn` and `PseudoMapTrainer`
- the editable install

Update those paths before you remove the symlink.
