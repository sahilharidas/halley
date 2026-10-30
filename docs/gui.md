# The GUI

Halley comes with a GUI: pages in your web browser, served by a small web server that runs on
your own computer. In it you can build a model, run it, read what a run wrote, explain any value
and compare two runs, without writing the Python that the [tutorial](tutorial/index.md) writes.
Nothing on its pages is computed by anything but the engine: a run made in the GUI is the run
the same call makes in Python, with the same run id.

## Starting it

```bash
python -m halley.gui
```

`halley-gui` does the same. The command starts the server, prints the address to open and opens
it in your default browser. It needs nothing beyond Halley's own dependencies. The runs it makes
go to a `runs` folder in the folder you start it from; `--runs` names another. Ctrl+C stops the
server; a build or run that has already started finishes first, and the command says it is
waiting for it.

| Option | What it does |
|---|---|
| `--runs FOLDER` | The folder whose run directories the GUI lists and where its runs go |
| `--model FILE` | A Python file of models to load at the start; give it once per file |
| `--port N` | The port to listen on, 8765 by default, or a free one when that is taken |
| `--host ADDRESS` | The address to listen on, 127.0.0.1 by default: this computer only |
| `--allow-host NAME` | A further name another computer reaches this one by; give it once per name |
| `--no-browser` | Print the address without opening it |

The address carries a token, a random string after `#token=`: the server answers only a page
that holds it, and only a request that names the server's own address, so another site open in
the same browser cannot use the GUI. Open the address the command printed; if you close the tab,
open it again from there.

## Models

The Models page lists the reference products that ship with Halley and the IFRS 17 stages. A model
of your own lives in a Python file: give its path in the field under the list, and the GUI imports
it as the module named after the file (`term.py` as `term`), with its folder on the import path.
Importing a file runs its top-level code, as `import term` would, so a file that also runs its
models keeps those runs under `if __name__ == "__main__":`. After you edit a model file, or a module
of your own one imports (one a formula imports included), the page says the files have changed;
reload them and build again. **Reload the files** runs every loaded file again, in the order you
loaded them, with every module of your own they import, each once, as a new Python process importing
them would; every earlier build is then a build of the files as they were. A file saved again within
the second its cached bytecode records, at the same size, which Python's cache cannot tell from the
file it was made from, has that cache taken out first, so the reload runs the file as it is now. A
loaded file that is no longer where you loaded it from (renamed, moved or deleted) is let go, with
its models, and the page says so. A reload is refused while a build, check or run is running; reload
when it has finished. A load or a reload that ends once you have gone to another page leaves that
page as it is, and says so in the tray (below). A file that fails to load takes out what its import
brought in of yours, so the next load runs it afresh, unless a build, check or run was going
meanwhile, which may use it: then it stays, and an edit to it shows as a change, which a reload
settles. A file that was imported before you loaded it, by an import the GUI did not see (by a
program that started the GUI, say), shows as changed until a reload, since nothing tells which
version of it that import ran; so does a file saved while it ran. A file whose name another module
already has (`json.py`, say) is refused, since a run with worker processes would import the other
one.

A model is built first. The build traces its formulas, checks their units, time references and
order, and generates the code that runs them, as `halley.build` does. Choose the backend, NumPy (the
reference) or Numba, and set the build-time parameters the class declares, each in a field of its
type: an enum's members are a list to choose from, a flag's are typed joined by `|` (`R|W`), a date
is a date, and a tuple or any other type is written as JSON, a tuple as a list, with an enum's
member by its name (or as Python writes it, `Basis.BEST`, but never as a label its class's own
`__str__` or `__repr__` gives) and a date in ISO form inside it (`["BEST", "2027-01-01"]`), a list's
items and a dict's keys too (a sequence in a key as a tuple). A setting a run's manifest cannot
record exactly builds, with a warning, since the run cannot be rerun from its directory: a dict that
is not the class's own default and whose keys are not in the order the manifest writes them (sorted
as text, by what `str()` gives for each key: `10` before `2`, and an int or str enum's member by its
value, not its name; the warning gives the order), a tuple given where the type is not a tuple (a
list's, say), which the manifest writes as a list, a NumPy number, an object the manifest records as
its repr; and an enum's member its class's own `__repr__` writes as it writes another, which a rerun
cannot tell apart, so the run cannot be rerun from its directory unless the member makes no
difference to the model. A dict recorded as its default is recorded may be the default or that dict
sorted, and a rerun tries each such dict both ways, all at their defaults and all sorted first, up
to 64 mixes of them; a build that gives a mix beyond those warns of it. A parameter you leave as it
is takes the class's default; a check box for a parameter without a default builds False when it is
left clear. Once the model is built, the form shows that build's settings and **Build again** starts
from them: a field you leave as it is keeps the build's own value, so the same settings build the
same IR hash. After a reload, a build of the files as they were carries its settings to the class as
it is now: an enum's member by its name, a setting it left at the default as the class's default
now, and one that held an object of a class of the file's as the default. A mistake in the model is
shown with the line of your file it points at and the lines around it; a build warning, such as an
unused variable, is shown beside the build.

A built model can run. Its inputs are a reference product's own example (reference case 1 for the
term assurance, the annuity specification's worked example for the annuities, lifelib's inputs for
`BasicTerm` in a checkout of the repository), or files: one for the model points and one per table
and scenario input the model declares, each Parquet or CSV. A CSV file is read with pyarrow's CSV
reader, as the [tutorial](tutorial/your-own-data.md) reads one. A relative path starts from the
folder you started the GUI in. Set the valuation date and, if you need them, the horizon, the batch
size, the worker processes, a label and the variables to store. **Check the inputs** loads them and
runs every check the loader makes, without projecting; **Run** projects every policy and writes a
new run directory under the runs folder, then opens it. The job that is running shows in the tray in
the page's bottom corner. A build, run or check that ends when its page is no longer showing it (you
went elsewhere, or came back and the page was drawn again) says so there, and the page has room at
its end for the tray, so nothing stays under it. When the job failed, or for a check's result, its
chip opens what the job said, with a link back to the page it was started on; it stays, so you can
open the report again, until you clear the tray, or for two minutes after you last open it while it
is on screen, but never while the pointer or the keyboard's focus is in the tray. A chip that comes
meanwhile comes above the chips there, which stay where they are; **Clear** stays at the tray's foot
however many chips it holds; pressed from the keyboard, it gives the focus back to the control it
came to the tray from. While a report is open the page beneath it is not drawn again: a build, run,
check, load or reload that ends then says so in the tray, and in its card, with a link that shows
what the card would have shown; a reload's line takes the place of the warning that the files
changed, and a run's card takes another name for the next run.

Below the build, the model's variables are listed with their units, roles, timing, recursion
direction and formulas; open one to see its source, what it reads and what reads it. The other
tabs show the inputs the model declares, its invariants and the code the build generated.

## Runs

The Runs page lists every run directory in the runs folder and the runs this session made or
opened, newest first. **Open a run directory** opens any other directory a run wrote, wherever it
is; a relative path starts from the folder you started the GUI in, then from the runs folder. A
run's page has these tabs:

- **Overview**: the model and its source, the settings, the backend and the run id, from the
  manifest. A link verifies every file the manifest records a hash for.
- **Charts**: a variable over the months for the policies you pick (each keeps its colour while
  others come and go), and summed over every policy. Each chart has a table view of its values.
- **Policies**: every policy's value of a variable at a month, beside its model point fields,
  largest first.
- **Explain**: a value of one policy at one month, its formula, where the formula is defined and
  the value of every input it read, as `halley.explain` gives them. **Explain** beside an input
  opens that input in turn; the path you took is shown above the value. The value is also
  recomputed from the stored inputs by Halley's interpreter, independently of the generated code,
  and the page says whether the two agree bit for bit (for a Numba run, within the 1e-9 the
  backends are held to).
- **Checks**: the invariants the run checked. A run that exists passed them all: a failing check
  stops a run before it writes anything.
- **Inputs**: the copy of every input the run read, each read back and checked against the hash
  the manifest records.
- **Files**: the manifest, the invariant results and the generated code.

Explain needs the model and the inputs a run was made with, which a run read back from its
directory does not hold. For such a run the Overview offers **Rerun it from its own inputs**: the
GUI builds the model the manifest names, with its recorded build-time settings, and runs it on
the directory's own copies of its inputs with its recorded settings, into the `reruns` folder
under the runs folder. The page then says whether the rerun has the run's id and the same values
bit for bit, and the rerun can be explained. An aggregation run made by
`halley.aggregation.run_aggregation` is replayed from its directory the same way
(`replay_aggregation`). A run made by a script run as `python file.py` names its model
`__main__.<class>`, which no import of the file can give it, so it cannot be rerun: keep a model
in a module of its own and import it in the script.

## Comparing two runs

Select two runs on the Runs page and compare them. The page lists everything a run id covers,
the model's IR, every input's hash, the settings that change the numbers and the backend, and
says which of them the two runs differ in. For two runs of the same policies over the same
months it lists the largest difference in every variable both hold, largest first, as
`halley.diff` finds them; open a row to see the two runs' values of that policy on one chart.

## What it does not do

- A build or run cannot be stopped once it has started: the engine has no way to stop a run part
  way. One waiting behind another can be cancelled, from its own page or from the tray.
  Builds and runs take turns, one at a time.
- The GUI does not run an aggregation stage, which runs on projection run directories through
  `halley.aggregation.run_aggregation`; it reads a stage's runs, replays them and explains them.
- A run's page reads the whole run into memory, as `halley.load_run` does, so a very large run
  is slow to open. The session holds the runs it has used most recently, up to eight it made,
  six it read back and about 1.5 GB in all, and lets go of the others, which stay on disk. A
  run it made and let go of is explained after a rerun, which its Explain tab offers.
- The GUI does not edit models. A model lives in a file, which you edit in your editor and the
  GUI imports, since the engine reads the file to point its messages at your lines.
- The GUI follows the modules of your own that Python's import system imports; one that puts an
  object, a module object or another module in its own place in `sys.modules`
  (`sys.modules[__name__] = _Basis()`) by the file that import ran, and one that runs another file
  as itself (`importlib.util.spec_from_file_location(__name__, ...)`) by both files. A module
  another file's code puts in a module's place in `sys.modules` shows that module as changed until a
  reload. One a file runs by its path without putting it in `sys.modules` runs again at a reload,
  but an edit to it does not show as a change; one imported from a zip archive is taken as an
  installed package's, and a reload does not import it again. A model file that puts something else
  in its own place cannot be loaded, nor reloaded, once another loaded file imported it, and the
  refusal says so. An edit to a file of yours that Python's path finder would find for a module that
  a finder of your own serves from another file shows as a change, which a reload settles.
- The server listens on this computer only. `--host` with one of this computer's addresses, or
  `0.0.0.0` for all of them, opens it to a network: another computer then opens
  `http://<this computer's address>:<port>/#token=...`, by any of this computer's addresses, or
  by a name given with `--allow-host` (the computer's own name is known already). Anyone who can
  reach the address and holds the token can then run Python code on this computer through the
  GUI.
