



. simulation.sh CNN DD 100 ; wait ; \
. simulation.sh CNN REDDIT-BINARY 100 ; wait ; \
. simulation.sh BrainCNN DD 100 ; wait ; \
. simulation.sh BrainCNN REDDIT-BINARY 100 ; wait ; \
. simulation.sh CNN subset1 100 ; wait ; \

. simulation.sh CNN poisson 100 ; wait ; \
. simulation.sh CNN new_poisson 100 ; wait ; \
. simulation.sh BrainCNN subset1 100 ; wait ; \
. simulation.sh BrainCNN poisson 100 ; wait ; \
. simulation.sh BrainCNN new_poisson 100


