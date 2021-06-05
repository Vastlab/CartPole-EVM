# CartPole-EVM

UCCS Darpa SAILON TA2 client for carpole using lookahead controller with similarity based on expected and actual next
state with an EVM based on state difference as novelty detector.

Authors: Jono Schwan, N. Windesheim, T. Boult

spec-file.txt has the packages from the necessary conda environment.

# Usage Instructions

```git clone --recursive https://github.com/Vastlab/CartPole-EVM.git```

Clone the repository the vast repo is added as a submodule of CartPole-EVM. 

Once the repo is cloned it is ready to be ran using docker-compose.

```
docker-compose -f dc-uccs-TA2.yml build
docker-compose -f dc-uccs-TA2.yml up -d --scale uccs-ta2-cartpole-0.6.2=1
docker-compose -f dc-uccs-TA2.yml logs -f --tail=5
```

Once the testing episodes begin you can run more docker containers to speed up testing in parallel.

```
docker-compose -f dc-uccs-TA2.yml up -d --scale uccs-ta2-cartpole-0.6.2=5
```

When the testing is completed stop the docker containers.

```
docker-compose -f dc-uccs-TA2.yml down
```
