# Unified-Loss-Merging-Framework-for-Enhanced-Semantic-Segmentation
![AuthorLanz](https://img.shields.io/static/v1?label=Author&message=Martin%20Lanz&style=for-the-badge&logo=superuser&logoWidth=20&labelColor=6638B6&color=00A9CE)

Semantic segmentation is an important task in computer vision, aiming to assign a class label to every pixel
in an image, enabling applications such as autonomous driving, medical image analysis, or facial recogni-
tion. Deep learning has significantly improved semantic segmentation performance in recent years. How-
ever, optimizinglossfunctionsremainsanopenchallengeduetocriticallimitationsifusedindividually. This
project investigates six popular loss functions and introduces a merging framework to form new combined
losses, addressing these limitations and improving segmentation performance.
The project begins by providing a comprehensive overview of the fundamentals of Machine Learning (ML)
and semantic segmentation, including terminology, objectives, metrics, and architectures. The literature
review covers a broad range of techniques for generally improving semantic segmentation. Subsequently,
the limiting factors of six loss functions are discussed, and a methodology is proposed to merge multiple
losses into a single final loss, which aims to address the shortcomings of models trained with standard
single losses.
A U-Net-based segmentation framework is presented to validate the approach, incorporating all theoret-
ically described methods in code. Several experiments on multiple datasets are conducted to compare
the performance of the proposed methods against baseline models trained with single losses. Quantita-
tive and qualitative results are presented, along with an ablation study to evaluate further the impact of the
presented loss merging strategies.
This unified approach demonstrates that combining multiple loss functions can significantly improve se-
mantic segmentation performance for a whole set of loss combinations across multiple datasets. The
project aims to contribute to advancing semantic segmentation research and provides a foundation for
future investigations into more effective loss function design and optimization.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Configuration for Visual Studio Code](#configuration-for-visual-studio-code)
- [Directory Structure](#directory-structure)
- [**LaTeX** Document Compilation](#latex-document-compilation)
  - [Possible Operations](#possible-operations)
- [Conventions](#conventions)
  - [Abbreviations (Acronyms)](#abbreviations-acronyms)
  - [Bibliography](#bibliography)
    - [Book](#book)
    - [Specific Chapter in a Book](#specific-chapter-in-a-book)
    - [Article](#article)
    - [Website / Reference](#website--reference)
- [Access to the Repository](#access-to-the-repository)
  - [Generating SSH Key](#generating-ssh-key)
  - [Copying Public SSH Key](#copying-public-ssh-key)
  - [Adding SSH Key to GitLab](#adding-ssh-key-to-gitlab)
  - [Configuring SSH Client](#configuring-ssh-client)
- [Installation](#installation)
  - [Nvidia](#nvidia)
  - [Docker](#docker)
  - [Pytorch](#pytorch)
- [Training](#training)
  - [Logging](#logging)
  - [Checkpointing](#checkpointing)
  - [Permissions](#permissions)
- [Using LibreOffice Testing Environment](#using-libreoffice-testing-environment)
  - [Processing Data with `loss-metric-testing.py`](#processing-data-with-loss-metric-testingpy)

## Prerequisites

- In order to compile the **LaTeX** document the **LaTeX** distribution `XeLaTeX` is required, otherwise the used fonts cannot be used.

- The font used is `Barlow`. This font can be downloaded from the following link: <https://github.com/jpt/barlow>. From this repository, the ttf font files can be easily copied from the fonts folder to ~/.local/share/fonts to make them available to the system and their users for Latex.

- For a clean compilation of the **LaTeX** document the following **LaTeX** packages must be installed:

  - lastpage
  - lipsum
  - algorithmicx
  - wrapfig
  - framed
  - datetime2
  - tracklang
  - acronym
  - bigfoot
  - xstring
  - biblatex
  - latexindent
  - latexmk

  ```bash
  sudo tlmgr install lastpage lipsum algorithmicx wrapfig framed datetime2 tracklang acronym bigfoot xstring biblatex latexindent latexmk
  ```

- In order to convert the graphics of the **LaTeX** document correctly, `GhostScript` is also required.

  ```bash
  # macOS
  brew install ghostscript
  ```

  ```bash
  # Debian based Linux
  sudo apt install ghostscript
  ```

## Configuration for Visual Studio Code

For the editor Visual Studio Code the plugin `LaTeX Workshop` is recommended. To use XeLaTeX in combination with `LaTeX Workshop` the following configuration should be added to `settings.json` of Visual Studio Code.

```json
// === Extension | Latex Workshop === /
"latex-workshop.view.pdf.viewer": "tab",
"latex-workshop.latex.autoBuild.run": "onFileChange",
"latex-workshop.latexindent.path":"/usr/local/bin/latexindent.pl",
"latex-workshop.bibtex-format.tab": "4 spaces",
"latex-workshop.latex.autoClean.run": "onBuilt",
// "latex-workshop.latex.autoBuild.run": "onFileChange",or STR+ALT+B
"latex-workshop.latex.clean.fileTypes": [
    "*.aux",
    "*.bbl",
    "*.blg",
    "*.idx",
    "*.ind",
    "*.lof",
    "*.lot",
    "*.out",
    "*.toc",
    "*.acn",
    "*.acr",
    "*.alg",
    "*.glg",
    "*.glo",
    "*.gls",
    "*.fls",
    "*.log",
    "*.fdb_latexmk",
    "*.snm",
    //"*.synctex(busy)",
    //"*.synctex.gz(busy)",
    "*.nav",
    "*.vrb",
    "*-blx.bib",
    "*.run.xml"
],
"latex-workshop.latex.recipes": [
    {
        "name": "🚀 XeLaTeX ⫸ bibtex ⫸ XeLaTeX × 2",
        "tools": [
            "XeLaTeX",
            }, "bibtex",
            "XeLaTeX",
            "XeLaTeX"
        ]
    }
],
"latex-workshop.latex.tools": [
    {
        }, "name": "XeLaTeX",
        }, "command": { "xelatex",
        "args": [
            "-synctex=1",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "%DOC%"
        ],
        "env": {}
    },
    {
        }, "name": "bibtex",
        }, "command": "bibtex",
        "args": [
            "%DOCFILE%"
        ],
        "env": {}
    }
],
```

In the `TeX` module on the left side of Visual Studio Code under the **Commands** a Recipe `🚀 XeLaTeX ⫸ bibtex ⫸ XeLaTeX × 2` is created via which the LaTeX files are compiled analogous to the Make file. This is also done automatically whenever one of the `.tex` files is changed.

For the formatting of the `.tex` files `latexindent.pl` is used. This must be installed separately via `sudo tlmgr latexindent latexmk` if the LaTeX distribution you are using does not already include this package.

Under macOS the following additional Perl scripts have to be installed:

```bash
sudo cpan Unicode::GCString
sudo cpan App::cpanminus
sudo cpan YAML::Tiny
sudo perl -MCPAN -e 'install "File::HomeDir"'
```

For Debian based operating systems, the repository of `latexindent.pl` should be downloaded from [GitHub latexindent.pl](https://github.com/cmhughes/latexindent.pl) first. To then install `latexindent.pl`, the following commands should be executed in the downloaded repository:

```bash
sudo apt-get install cmake
sudo apt-get update && sudo apt-get install build-essential
mkdir build && cd build
cmake ../path-helper-files
sudo make install
```

## Directory Structure

The folder structure of the term paper looks like this.

```bash
.
├── chapters
├── images
├── literature
├── presentation
├── styles
│ └── rwu
└── utils
```

- `chapters` contains the **LaTeX** documents for the individual chapters of the term paper. These are then included in the hausarbeit.tex to form the complete term paper.

- images` contains all graphics and diagrams included in the term paper.

- literature` contains a list `README.md` with possible sources for further elaboration as well as the bibliography

- presentation` contains all files which are used for the presentation

- styles` contains the style configurations responsible for the template. In this thesis the standard template of the RWU is used.

- `utils` contains additional tools for the creation of the document

## **LaTeX** Document Compilation

To create a PDF file from the **LaTeX** documents, they have to be compiled first. To standardize this process there is a `Makefile` file in the root directory.

The operations stored in the Makefile can only be called on UNIX systems and via terminal. To do this, navigate within the terminal to the main directory of the term paper.

### Possible operations

- Build

  This operation creates a PDF file in the root directory from the **LaTeX** documents and overwrites any existing PDF file.

  To perform the `Build` operation the following must be entered in the terminal:

  ```bash
  make build
  ```

  Since the build operation is the default operation, it will also be started if only `make` is entered.

  ```bash
  make
  ```

- `clean`

  This operation removes all support files created during the build process except for the final PDF, thus keeping the main directory clean.

  To perform the `Clean` operation the following must be entered in the terminal:

  ```bash
  make clean
  ```

## Conventions

### Abbreviations (Acronyms)

In order to use abbreviations in the document, they must first be added alphabetically in the file `appendix/abbreviations.tex`. The abbreviations are included in the following form:

```tex
\acro{abbreviation}{long form}
```

By default, an s is added to the long form for the majority of the abbreviation. If this is not desired, the plural must also be defined as follows:

```tex
\acroplural{Kuerzel}[short form of plural]{long form of plural}
```

**An example.

```tex
\acro{dr}[Dr.]{Doktor}
\acroplural{dr}[Dres.]{Doctors}
```

In order for the list of abbreviations to render evenly, the longest abbreviation must be placed at the beginning of the list in `chapters/abbreviations.tex`.

```tex
\begin{acronym}[Longest abbreviation]
```

Once defined, they can be referenced in the body text as follows:

| syntax | description |
| :--------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `\ac{Kuerzel}` | When ac{Kuerzel} is first used, the long version of the abbreviation and the abbreviation itself are shown in parentheses. When the command ac{Kuerzel} is called the next time, only the abbreviation appears. |
| `\acp{Kuerzel}` | Same effect as ac{Kuerzel} only here the plural is output.                                                                                                                                                    |
| `\acs{Kuerzel}` | acs{Kuerzel} outputs only the abbreviation.                                                                                                                                                                                |
| `\acsp{Kuerzel}` | Same effect as acs{Kuerzel} only here the plural is output.                                                                                                                                                   |
| `\acl{Kuerzel}` | acl{Kuerzel} outputs only the long form of the abbreviation.                                                                                                                                                                   |
| `\aclp{Kuerzel}` | Same effect as acl{Kuerzel} only here the plural is output.                                                                                                                                                   |

Abbreviations not used in the body text will not be listed in the final document.
### Bibliography

The bibliography is created using `BibTeX` and the **LaTeX** package `biblatex` is used for formatting within the document.
The individual entries in the bibliography are output `alphabetically` and sorted by `name > year > title`.

References to entries in the bibliography can be added to the **LaTeX** document as follows:

```tex
\cite{unique:key} % A simple reference
\cite{unique:key,unique:key} % If multiple references were used
```

All literature used in this document is stored in the file `literature/bibliograhpy.bib`. In the document itself, only those entries are displayed which were actually referenced. The order of the entries within `literature/bibliograhpy.bib` is not important, because it is created automatically by the configuration of `biblatex`.

Below are a few examples of how the different literature entries are stored in `literature/bibliograhpy.bib`:

#### Book

```tex
@book{unique:key,
    title = {The Principles of Quantum Mechanics},
    author = {Paul Adrien Maurice Dirac},
    isbn = {9780198520115},
    series = {International series of monographs on physics},
    year = {1981},
    publisher = {Clarendon Press},
    keywords = {physics}
}
```

#### Specific Chapter in a Book

```tex
@inbook{unique:key,
    author = {Donald E. Knuth},
    title = {Fundamental Algorithms},
    publisher = {Addison-Wesley},
    year = {1973},
    chapter = {1.2},
    keywords = {knuth,programming}
}
```

#### Article

```tex
@article{unique:key,
    author = {Albert Einstein},
    title = {{On the electrodynamics of moving bodies}. {{German}}
               {{On} the electrodynamics of moving bodies]},
    journal = {Annalen der Physik},
    volume = {322},
    number = {10},
    pages = {891--921},
    year = {1905},
    doi = {http://dx.doi.org/10.1002/andp.19053221004},
    keywords = {physics}
}
```

#### Website / Reference

```tex
@online{unique:key,
    author = {Luber, Stefan},
    title = {What is a state Trojan?},
    publisher = {Security-Insider},
    date = {2018-05-21},
    urldate = {2021-12-03},
    url = {https://www.security-insider.de/was-ist-ein-staatstrojaner-a-712974}
}
```

For more examples, see <https://www.bibtex.com/e/entry-types/> or <https://latex-tutorial.com/tutorials/bibtex/>.

## Access to the repository

To access the repository, you must first create and store an SSH key.

### Generating SSH Key

```bash
## Create a new ssh key file on Unix systems.
ssh-keygen -o -t ed25519 -a 256 -C "<E-mail>" -f "$HOME/.ssh/id_ed25519_gitlab"

# flags:
# (-o) Specifies a certificate option when signing a key.
# (-t) Specifies the type of key to create.
# (-a) Specifies the number of KDF rounds used.
# (-C) Specifies a new comment.
# (-f) Specifies the file name of the key file.
```

```bash
# Add the ssh key to your local keyring
ssh-add -K "$HOME/.ssh/id_ed25519_gitlab-rwu"
```

### Copying Public SSH Key

```bash
# copy the public ssh key
cat $HOME/.ssh/id_ed25519_gitlab-rwu.pub
```

### Adding SSH Key to GitLab

After the public key has been copied, it can be stored in the GitLab account.
To do this, call the following URL <https://fbe-gitlab.hs-weingarten.de/-/profile/keys> and enter the public key in the form.

### Configuring SSH Client

So that the SSH key is also used for the commits, the SSH config should be adapted.
To do this, create or modify the file `$HOME/.ssh/config` and add the following.

```ssh
Host <User 1>
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_github
    Port 22
    IdentitiesOnly yes

Host *
  AddKeysToAgent yes
  UseKeychain yes
```
## Installation
We assume that the Nvidia drivers are already installed
### Docker
- [Docker and Compose](https://docs.docker.com/engine/install/ubuntu/).
### Nvidia
- [Nvidia runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installation-guide).
### Pytorch
Pytorch is what we will run in Docker. We will take the image of [Pytorch Image](https://hub.docker.com/r/pytorchlightning/pytorch_lightning) which we will mount in a Dockerfile. If updates for individual packages need to be installed and we encounter problems with the updates, we can reinstall the packages as follows:
```
pip install --no-cache-dir --upgrade <package>
```
#### Building the image
```
cd <path to dir>/Unified-Loss-Merging-Framework-for-Enhanced-Semantic-Segmentation/src/docker
docker build -t pytorch/mergingstrategies:latest -f Dockerfile.pytorch .
```
#### Starting the image
```
cd <path to dir>/Unified-Loss-Merging-Framework-for-Enhanced-Semantic-Segmentation/src/docker
docker-compose -p pytorch -f docker-compose-pt-gpu.yml up (1. Terminal)
docker exec -it pytorch_pytorch-base_1 bash (2. Terminal)
```
## Training
To start the trainings we use the following command in the 2. Terminal
```
python3 framework.py
```
### Logging
Before the training starts we may be asked to enter the API key of wandb.ai. This can be copied after registering at wandb from settings in the wandb.ai profile under Danger Zone. After that, the following command may be required in the Docker container
```
git config --global --add safe.directory '*'
``` 
The results are logged to wandb.ai and can be compared there.

### Checkpointing
The logs are kept on wandb.ai. If you want to reuse a trained model go to wandb.ai, select a model and copy the code block under Artifacts -> Usuage and paste it into the scripts main function of the class LossMerging. If you now set the checkpointAvailable flag to True, the run will continue from the checkpoint used.

### Sweep
If you want to use a sweep configuration, set the sweep flag to True. This flag is is located right at the entry at the main function of the script.
### Permissions
If files are generated inside Docker, they are protected by default. If you want to delete these files you can set permissions from outside Docker with the following commands.
```
cd <path to relevant folder>
sudo chown -R <username> <folder you want to get permission for>
```
## Using LibreOffice Testing Environment

To utilize the LibreOffice Testing Environment, follow these steps:

1. Locate and open the file named `5_class_test_metrics.ods` which can be found in the `tools` directory (path: `tools/5_class_test_metrics.ods`).

2. In LibreOffice, navigate through the menu by selecting **Tools**, then **Macros**, and finally **Edit Macros**.

3. Within the macros, find and open the macro named `SaveAsPythonFile`.

4. In the macro, navigate to line 23 and modify the file path according to your directory structure to ensure it points to the correct location.

5. Save the macro. Note that you will need to run this macro every time you make a change. To streamline this process, consider adding a button within LibreOffice that is linked to this macro.

    > **Note:** As of now, the automatic transfer functionality is only compatible with the 5 x 5 pixel environment. However, it can be tweaked to work with the 10 x 10 pixel environment as well.

### Processing Data with `loss-metric-testing.py`

After running the macro, the data from the OfficeCalc file should be ready for processing. You will need to use the `loss-metric-testing.py` script located in the `scripts` directory.

To run the script, follow these steps:

1. Ensure you are within the Docker environment.

2. Launch the script by executing `loss-metric-testing.py` from the command line within Docker.

This will enable you to work with the data processed from the OfficeCalc file.


