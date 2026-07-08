# Next Steps for Parcelmate

## End goal:

If we remove subnetworks then does model stay on task?

---

## How to do this:

- Give prefix to model for a domain and then does it appear to be drifting from that domain if we knockout a subnetwork?
- Take chunk of Reddit and give half of it to healthy model and then half to knockout model and take loss function analysis

### Projects (all done from prefixes):

Ability to maintain persona (on bigger models)

- Tell it it’s a 38 yro conservative white guy - generations should imitate the demographics - knockouts maybe cause generations to stray from demographic persona?

Are these really controlling subnetworks and if so what are they controlling and how?

- Test in domains with prefixes

For a given number of neurons for a subnetwork - run a baseline to knockout same number of neurons in a different subnetwork

- Basically saying is the problem selective in some way and we need to control 
- Fixing weights to zero for subnetwork is easy - better knockout is “mean out”:
  - grab sample of text across all genres, compute average activation, clamp weights to that value

### Baselining and mean out:

- First design in code is suboptimal
  - need separate analyses for each network that emerges from parcellation
  - knockout each network individually
- For each network connectivity - do main knockout for the number of neurons
- 

---

### What is Baselining:

- Baseline is same knockout and neurons but with random selection of neurons in remaining set of neurons (not the ones that you originally knocked out)

### Why we are doing this approach:

- Knocking out from baseline from connectivity vs just knocking out neurons in general
  - If we are going to make claims that there’s some function that localizes to these neurons, then we need to compare when not localized

