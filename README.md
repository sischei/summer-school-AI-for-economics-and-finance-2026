<p align="center">
<img src="screens/SummerSchool_Torino_2026.png" width="800px"/>
</p>


Summer School on [AI for Economics and Finance](https://sites.google.com/carloalberto.org/deeplearningsummerschool2026), held at [ESOMAS](https://www.esomas.unito.it/do/home.pl), August 24 - 26, 2026, at the University of Torino, followed by a [conference](https://sites.google.com/carloalberto.org/deeplearningsummerschool2026/conference-program) on August 27 - 28, 2026.


## At a glance

| | |
|------|------|
| **Summer school** | Monday to Wednesday, August 24 - 26, 2026 |
| **Conference** | Thursday to Friday, August 27 - 28, 2026 |
| **Venue** | Classrooms 7 and 8, SME building, Corso Unione Sovietica 218bis, 10134 Torino |
| **Format** | Interactive, workshop-like lectures with hands-on coding exercises |
| **Computing** | Python on the [Nuvolos](https://nuvolos.com) cloud, no local setup required |
| **Website** | [sites.google.com/carloalberto.org/deeplearningsummerschool2026](https://sites.google.com/carloalberto.org/deeplearningsummerschool2026) |
| **Contact** | <deeplearningconf26@gmail.com> |


## About the summer school

The school equips Ph.D. students and researchers in economics and finance with the
computational tools reshaping the field, across two complementary frontiers: solving and
estimating dynamic stochastic models, and using LLMs and NLP for empirical work.

**Days 1 and 2: deep learning for dynamic stochastic models**

* **Deep Equilibrium Nets and deep surrogates**: solving dynamic stochastic models, then using surrogates to *estimate* them.
* **Heterogeneous agent models**: DeepHAM and structural reinforcement learning.
* **PDEs and continuous time**: physics-informed neural networks, and deep learning for continuous-time frameworks.

**Days 2 and 3: large language models and sequence modeling**

* **Foundations**: from RNNs and LSTMs to the Transformer.
* **The LLM revolution**: pre-training, BERT and GPT, scaling laws, emergent abilities.
* **Domain adaptation**: LoRA for financial and climate intelligence; RLHF/DPO alignment.
* **Frontiers**: retrieval-augmented generation, agentic LLMs, applications in finance.

Coding is in [Python](http://www.python.org) with [scikit-learn](https://scikit-learn.org/),
[PyTorch](https://pytorch.org/) and [Hugging Face Transformers](https://huggingface.co/docs/transformers).


## Preparing for the school

### Prerequisites

* Basic econometrics.
* Basic programming in Python (see [this link to QuantEcon](https://python-programming.quantecon.org/intro.html) for a thorough introduction).
* A brief Python refresher is provided [under this link](python_refresher).
* A brief introduction to Jupyter Notebooks is provided [under this link](python_refresher/jupyter_intro.ipynb).
* Basic calculus and probability.

### Recommended reading

* **[Reading list (PDF)](Reading_List.pdf)**, the pre-reading and pre-coding to complete before arriving, plus core and background readings for every session ([LaTeX source](Reading_List.tex)).
* [Mathematics for Machine Learning](https://mml-book.github.io/), a good overview of the mathematical skills participants are expected to be fluent in.
* [Deep Learning](https://www.deeplearningbook.org/) (Goodfellow, Bengio, and Courville).

### Class enrollment on the [Nuvolos Cloud](https://nuvolos.cloud/)

* All lecture materials (slides, codes, and further readings) will be distributed via the [Nuvolos Cloud](https://nuvolos.cloud/).
* To enroll in this class, please click on the [enrollment key](https://app.eu1.nuvolos.cloud/enroll/class/9t_WlkNEcHk), and follow the steps.

### What is in this repository

| Folder | Contents |
|------|------|
| [`day1/`](day1), [`day2/`](day2), [`day3/`](day3) | One folder per lecturer, each with `slides/`, `readings/`, and `code/` |
| [`python_refresher/`](python_refresher) | Self-study notebooks on Python basics and Jupyter |
| [`Reading_List.pdf`](Reading_List.pdf) | Per-session reading list, with the [LaTeX source](Reading_List.tex) alongside it |


## Schedule

> **Tentative, to be confirmed.** The final program will be published on the [summer school website](https://sites.google.com/carloalberto.org/deeplearningsummerschool2026/course-program).

### [Day 1](day1), Monday, August 24th, 2026

| **Time** | **Main Topics** | **Lecturer** |
| --- | --- | --- |
| 08:30 - 09:00 | Registration | |
| 09:00 - 09:10 | [Welcome by the organizers](day1/Trojani/slides) | Trojani |
| 09:10 - 10:30 | [Introduction to Deep Learning and Deep Equilibrium Nets (part I)](day1/Scheidegger/slides/01_Intro_to_DeepLearning.pdf) | Scheidegger |
| 10:30 - 11:00 | Coffee Break | |
| 11:00 - 12:30 | [Deep Equilibrium Nets (part II)](day1/Scheidegger/slides/02_DEQN.pdf) | Scheidegger |
| 12:30 - 13:30 | Lunch Break | |
| 13:30 - 15:00 | [Deep Surrogates to Estimate Dynamic Models](day1/Scheidegger/slides/03_Deep_Surrogates.pdf) | Scheidegger |
| 15:00 - 15:30 | Coffee Break | |
| 15:30 - 17:00 | [Solving heterogeneous agent models with DeepHAM and structural reinforcement learning](day1/Yang/slides) | Yang |
| 19:00 - | Pizza Dinner | |

### [Day 2](day2), Tuesday, August 25th, 2026

| **Time** | **Main Topics** | **Lecturer** |
| --- | --- | --- |
| 09:00 - 10:30 | [Physics-informed Neural Nets (PINNs) for solving Partial Differential Equations](day2/Scheidegger/slides/04_PINNs.pdf) | Scheidegger |
| 10:30 - 11:00 | Coffee Break | |
| 11:00 - 12:30 | [Deep Learning for continuous-time models](day2/Yang/slides) | Yang |
| 12:30 - 13:30 | Lunch Break | |
| 13:30 - 15:00 | [Gaussian Processes for Dynamic Portfolio Optimization](day2/Trojani/slides) | Trojani |
| 15:00 - 15:30 | Coffee Break | |
| 15:30 - 17:00 | [Foundations of Sequence Modelling: From RNNs to Transformers](day2/Leippold/slides) | Leippold |
| 19:00 - | Informal Aperitivo/Drinks (self-organized and self-funded gathering for an aperitivo or drinks). | |

### [Day 3](day3), Wednesday, August 26th, 2026

| **Time** | **Main Topics** | **Lecturer** |
| --- | --- | --- |
| 09:00 - 10:30 | [The LLM Revolution: Scaling Laws and Modern Architectures](day3/Leippold/slides) | Leippold |
| 10:30 - 11:00 | Coffee Break | |
| 11:00 - 12:30 | [Domain Adaptation: Building Financial & Climate Intelligence](day3/Leippold/slides) | Leippold |
| 12:30 - 13:30 | Lunch Break | |
| 13:30 - 15:00 | [Advanced Frontiers: From RAG to Reasoning and Agentic LLMs](day3/Leippold/slides) | Leippold |
| 15:00 - 15:30 | Coffee Break | |
| 15:30 - 17:00 | [An Introduction to Recursive Networks for conditional Asset Pricing](day3/Trojani/slides) | Trojani |

### The conference, August 27 - 28, 2026

Directly following the school, the *Deep Learning for Dynamic Stochastic Models* conference brings together scholars working on high-dimensional, nonlinear and stochastic models: invited talks, contributed presentations and panel discussions. See the [conference program](https://sites.google.com/carloalberto.org/deeplearningsummerschool2026/conference-program).

Thursday, August 27th, 2026:

| **Time** | **Main Topics** | **Speakers** |
| --- | --- | --- |
| 17:15 - 18:15 | [Panel discussion: Challenges and opportunities for finance research in the era of large language models and AI agents](https://www.carloalberto.org/event/challenges-and-opportunities-for-finance-research-in-the-era-of-large-language-models-and-ai-agents/) | Leippold, Oswald, Scheidegger, Trojani |

The panel is a side initiative of the summer school, organized by the ESOMAS Department and Collegio Carlo Alberto. It takes place at Collegio Carlo Alberto, Piazza Arbarello 8, which is *not* the summer school venue. Register [in person](https://www.eventbrite.it/e/challenges-and-opportunities-for-finance-research-in-the-era-of-large-llms-tickets-1994646487338?aff=oddtdtcreator) or [online](https://us02web.zoom.us/webinar/register/WN_WOilegCBQMmKmp751CI15Q).


## Practical information

### Venue

Classrooms 7 and 8, ground floor next to the main entrance of the SME building, Corso Unione Sovietica 218bis, 10134 Torino.

### Registration, fees, and deadlines

Registration, the fee structure, and the application deadlines are all handled on the [summer school website](https://sites.google.com/carloalberto.org/deeplearningsummerschool2026), which is the authoritative source for these details.

### Accommodation

Free choice; a few options near the venue:

* **Hotel Astor**, Piazza Tancredi Galimberti 12, 10134 Torino. Offers a **special rate for guests affiliated with the University of Turin** during the summer school period. Accepted participants receive the booking instructions by email; if you have not received them, please contact <deeplearningconf26@gmail.com>.

Other nearby hotels (no special rates):

* **Hotel Original**, Via Arturo Farinelli 4, 10135 Torino
* **B&B Città Giardino**, Google Plus Code: 2J8C+WP, Turin
* **B&B da Malù**, Google Plus Code: 2J9W+MR, Turin
* **Hotel Cristallo**, Corso Traiano 28/9, 10135 Torino
* **Hotel Gran Torino**, Via La Loggia 6, 10134 Torino

### Getting to Torino

Torino is served by two major railway stations, both well connected:

* **Torino Porta Susa**, high-speed trains (Frecciarossa, Italo) from cities such as Milan and Rome.
* **Torino Porta Nuova**, centrally located, with high-speed (Frecciarossa, Italo) and regional connections.

Torino Airport (TRN) is connected to both stations by train and shuttle bus (approximately 30 - 45 minutes); see [SAGAT](https://www.aeroportoditorino.it/) for details.

### Getting to the venue

From the city center, the most convenient option is **Tram #4**, which runs frequently and passes through Via Milano, Via Arsenale, Via Sacchi, and Porta Nuova station. It continues south along Corso Unione Sovietica and stops at **Montevideo**, directly in front of the venue.

Tickets and info:

* Standard ticket: €1.90, valid for 100 minutes on trams, buses, and the metro.
* Purchase from Tabacchi (tobacconists), newsstands, or via the GTT TO Move app.
* No ticket? You can tap your contactless bank card on the validation machine inside the tram or bus. This costs €2.00 and is also valid for 100 minutes.

### Support and contacts

* Organization, local assistance and any other question: <deeplearningconf26@gmail.com>
* Computing platform: Nuvolos Support, <support@nuvolos.cloud>


## People

### Lecturers

- [Markus Leippold](https://www.df.uzh.ch/en/people/professor/leippold.html) (University of Zurich and Swiss Finance Institute), <markus.leippold@df.uzh.ch>
- [Simon Scheidegger](https://sites.google.com/site/simonscheidegger/) (University of Lausanne and Grantham Research Institute, London School of Economics), <simon.scheidegger@unil.ch>
- [Fabio Trojani](https://www.unige.ch/gsem/en/research/faculty/all/fabio-trojani/) (University of Geneva and Swiss Finance Institute), <fabio.trojani@unige.ch>
- [Yucheng Yang](https://sites.google.com/site/yangyucheng1993/home) (University of Zurich and Swiss Finance Institute), <yucheng.yang@uzh.ch>

### Panelists

- [Markus Leippold](https://www.df.uzh.ch/en/people/professor/leippold.html) (University of Zurich and Swiss Finance Institute)
- [Florian Oswald](https://floswald.github.io/) (University of Turin and Collegio Carlo Alberto), Data Editor of the *Journal of Political Economy*
- [Simon Scheidegger](https://sites.google.com/site/simonscheidegger/) (University of Lausanne and Grantham Research Institute, London School of Economics)
- [Fabio Trojani](https://www.unige.ch/gsem/en/research/faculty/all/fabio-trojani/) (University of Geneva and Swiss Finance Institute)

### Co-academic directors

- [Simon Scheidegger](https://sites.google.com/site/simonscheidegger/) (University of Lausanne and Grantham Research Institute, London School of Economics)
- [Fabio Trojani](https://www.unige.ch/gsem/en/research/faculty/all/fabio-trojani/) (University of Geneva and Swiss Finance Institute)

### Local organizers

- [Andrea Gallice](https://sites.google.com/carloalberto.org/andreagallice/home-page) (University of Turin and Collegio Carlo Alberto), Head of the Local Organizing Committee
- [Alessandro Milazzo](https://www.esomas.unito.it/do/home.pl) (University of Turin and Collegio Carlo Alberto)
- [Alessandro Ricchiuti](https://www.esomas.unito.it/do/home.pl) (University of Turin)
