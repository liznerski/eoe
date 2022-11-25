import os
import os.path as pt
import random
from argparse import Namespace
from typing import Tuple, List

import numpy as np
import torch
from deap import base
from deap.tools.selection import attrgetter
from torch import Tensor
from torch.utils.data import Subset

from eoe.evolve.tree import Node, Tree, EvolNode, Individual
from eoe.training.ad_trainer import ADTrainer
from eoe.utils.logger import Logger


def match_samples(samples1: Tensor, samples2: Tensor) -> List[int]:
    """
    Used by :method:`mate_individuals` to match the two parents s.t. the OE images of them appear in pairs.
    That is, samples1[i] and samples2[i] are as close as possible.
    For this purpose, returns the argmax of the distances to all elements of the first list for each element of the second list.
    """
    distances = (
            samples1.flatten(1) - samples2.flatten(1).unsqueeze(0).repeat(samples1.size(0), 1, 1).transpose(0, 1)
    ).pow(2).sum(-1)

    def ms(d):
        n = d.size(0)
        if n == 1:
            return d[0, 0].item(), [0]
        rs = [ms(torch.cat([d[1:, :b], d[1:, b+1:]], dim=1)) for b in range(n)]
        rs = [(d[0, b].item() + r[0], [b] + [p if p < b else p + 1 for p in r[1]]) for b, r in enumerate(rs)]
        r = min(rs, key=lambda r: r[0])
        return r

    r = ms(distances)
    return r[1]


def init_individual(oeds: Subset) -> int:
    """ samples a random index from the OE dataset that can be used as a part of the OE subset """
    # init_individual is invoked at the start of the evolutionary algorithm. Since every dataset is actually a
    # torch.data.utils.Subset (see :class:`eoe.datasets.bases.TorchvisionDataset`), we here remember the
    # original indices of the Subset in oeds.valid_indices. For most OE datasets, these indices will cover the complete
    # dataset, but for some (e.g., ImageNet-30) they exclude some classes and thus the OE sample indices we randomly
    # select during the evolution need to be chosen according to valid_indices. For example, index "5" refers to the
    # 5th samples of the original Subset and not to the 5th samples of the complete dataset.
    # Again, for most OE datasets this doesn't make a difference.
    oeds.valid_indices = oeds.indices
    return np.random.randint(0, len(oeds.valid_indices))


def evaluate_individual(individual: object, trainer: ADTrainer, args: Namespace, toolbox: base.Toolbox) -> float:
    """
    Evaluates an OE subset (i.e., individual of the population) by executing the prepared trainer's run method.
    This, perhaps, iterates over multiple class-seed combinations. The mean AUC on the test set averaged over all
    class-seed combinations is the fitness of the OE subset.

    @param individual: the OE subset defined via a DEAP individual that contains a list of indices in the OE dataset.
    @param trainer: the prepared AD trainer.
    @param args: the experiment's configuration. Will be used to determine the classes and number of random seeds to
        iterate over.
    @param toolbox: The prepared DEAP toolbox that provides functions for mating, etc.
    @return: the fitness (i.e., mean test AUC).
    """
    if trainer.logger.active and len(os.listdir(trainer.logger.dir)) > 5:
        trainer.logger.deactivate()
    trainer.oe_limit_samples = list(map(toolbox.clone, individual))
    if trainer.ds is not None:
        # set indices to current individual so that OE dataset only uses that one
        trainer.ds.oe.train_set.indices = [trainer.ds.oe.train_set.valid_indices[i] for i in list(map(toolbox.clone, individual))]
    res = trainer.run(args.classes, args.iterations)[1]['mean_auc']
    if trainer.ds is not None:
        # set indices to all valid ones (usually all for OE and a subset using just one class for BCLF)
        trainer.ds.oe.train_set.indices = trainer.ds.oe.train_set.valid_indices
    return res


def mate_individuals(ind1: object, ind2: object, oeds: Subset, poolsize: int,
                     indp: float, oneofkbest: int) -> Tuple[object, object]:
    """
    Mates two individuals (~ OE subsets).
    If the OE subsets contain more than one OE image, mating works by randomly swapping OE images between the individuals.
    If the OE subsets consist of just one OE image, samples two random lists of candidates from the complete OE dataset.
    Similar to :method:`mutate_individual`, replaces the OE images in the subsets by narrowing the candidate lists
    down to a list of candidates with the least distance to BOTH OE images and then picking a replacement randomly from that.
    In this case, the resulting new individual contains an OE image that is somewhat "inbetween" the parents.
    Returns the new individuals, but the individuals are also updated in place.

    @param ind1: The first parent.
    @param ind2: The second parent.
    @param oeds: The complete Outlier Exposure (OE) dataset.
    @param poolsize: The size of the initial candidate list before narrowing down.
    @param indp: The chance to replace an OE image of the individual.
    @param oneofkbest: Defines the size of the candidate list after narrowing down.
    @return: The new individuals (the two children).
    """
    if len(ind1) == 1:  # mate single OE samples by searching for an image "inbetween"
        samples1 = torch.stack([oeds[id][0] for id in ind1])
        samples2 = torch.stack([oeds[id][0] for id in ind2])
        match_ids = match_samples(samples1, samples2)
        samples = torch.stack([torch.stack([samples1[a], samples2[b]]) for a, b in zip(range(samples1.size(0)), match_ids)])
        new_ids1 = [np.random.randint(0, len(oeds)) for _ in range(poolsize)]
        new_samples1 = torch.stack([oeds[id][0] for id in new_ids1])
        new_ids2 = [np.random.randint(0, len(oeds)) for _ in range(poolsize)]
        new_samples2 = torch.stack([oeds[id][0] for id in new_ids2])
        for n, double in enumerate(samples):
            if np.random.rand() < indp:
                distances = (double.unsqueeze(1) - new_samples1).pow(2).flatten(2).sum(-1).sum(0)
                val, arg = distances.sort()
                s = next(a for a in range(val.size(0)) if val[a] > 100)  # exclude self
                c = np.random.randint(s, s + oneofkbest)
                ind1[n] = new_ids1[arg[c]]
            if np.random.rand() < indp:
                distances = (double.unsqueeze(1) - new_samples2).pow(2).flatten(2).sum(-1).sum(0)
                val, arg = distances.sort()
                s = next(a for a in range(val.size(0)) if val[a] > 100)  # exclude self
                c = np.random.randint(s, s + oneofkbest)
                ind2[n] = new_ids2[arg[c]]
    else:  # mate OE sets by swapping elements
        for i in range(len(ind1)):
            if np.random.rand() < indp:
                tmp = ind1[i]
                ind1[i] = ind2[i]
                ind2[i] = tmp
    return ind1, ind2


def mutate_individual(ind: object, oeds: Subset, poolsize: int, indp: float,
                      oneofkbest: int) -> Tuple[object]:
    """
    Mutates an individual (~ OE subset). Samples a random list of candidates from the complete OE dataset.
    For each OE image in the individual, further narrows down the list of candidates by keeping only the `oneofkbest`
    candidates that have the least distance to the OE image. From this sublist, randomly selects a sample to replace
    the OE image in the individual.
    Returns the new individual, but the individual is also updated in place.

    @param ind: The DEAP individual defining the OE subset via a list of indices.
    @param oeds: the complete Outlier Exposure (OE) dataset.
    @param poolsize: The size of the initial candidate list before narrowing down.
    @param indp: The chance to replace an OE image of the individual.
    @param oneofkbest: Defines the size of the candidate list after narrowing down.
    @return: The mutated individual.
    """
    samples = torch.stack([oeds[id][0] for id in ind])
    new_ids = [np.random.randint(0, len(oeds)) for _ in range(poolsize)]
    new_samples = torch.stack([oeds[id][0] for id in new_ids])
    for n, sample in enumerate(samples):
        if np.random.rand() < indp:
            distances = (sample.unsqueeze(0) - new_samples).pow(2).flatten(1).sum(1)
            val, arg = distances.sort()
            s = next(a for a in range(val.size(0)) if val[a] > 100)  # exclude self
            c = np.random.randint(s, s + oneofkbest)
            ind[n] = new_ids[arg[c]]
    return ind,


def replace_individuals_randomly(individuals, oeds: Subset):
    """ replaces DEAP individuals by completely randomly sampling new ones from the OE dataset """
    for n, ind in enumerate(individuals):
        individuals[n] = np.random.randint(0, len(oeds.valid_indices))
    return individuals


def select_individual(individuals: List[object], k, tournsize, fit_attr="fitness", replace=False):
    """
    Based on the population (list of individuals ~ OE subsets), selects a new population of survivors according to the
    tournament selection rule. That is, each survivor is chosen by sampling `tournsize` many samples randomly from the population
    and picking the one with the best fitness. The survivor sampling works with replacement, so there can be duplicates in the
    new generation.

    @param individuals: The old population. A list of DEAP individuals, each containing a list of indices in the OE dataset.
    @param k: The number of survivors. Usually matches the length of the old population.
    @param tournsize: The number of candidates for selection (read above).
    @param fit_attr: The attribute that can be used to access the fitness of a DEAP individual.
    @param replace: Whether to also sample with replacement for the list of candidates.
    @return: the new population (i.e., the survivors).
    """
    chosen = []
    for _ in range(k):
        aspirants = [individuals[i] for i in np.random.choice(len(individuals), tournsize, replace)]
        chosen.append(max(aspirants, key=attrgetter(fit_attr)))
    return chosen


def evaluate(offspring: List[object], pop: List[object], gen: int, toolbox: base.Toolbox, history: dict, tree: Tree,
             oeds: Subset, logger: Logger):
    """
    Evaluates a population of offsprings by iterating over all OE subsets in it.
    In each iteration, performs a full training with the OE subset as OE,
    perhaps again iterating over multiple class-seed combinations. This depends on the configuration of the trainer.
    The fitness of the OE subset will be the mean AUC on the test set, averaged over all class-seed combinations.
    Stores the fitness in the genealogical tree.
    Also, creates an image file (e.g., png) with the concatenated OE images and logs that on the disk.
    The node in the genealogical tree will remember the path to this file.

    @param offspring: The population off offsprings that is to be evaluated.
    @param pop: the previous population (containing the parents). Once the evaluation is completed, this list will be
        overwritten with the offspring list. That is, the offsprings become the new population.
    @param gen: The current generation number; e.g., 0 for the first.
    @param toolbox: The prepared DEAP toolbox that provides functions for mating, etc.
    @param history: The history dictionary that stores information about the setup and all evaluated generations.
    @param tree: The genealogical tree with all the fitness values (i.e., mean AUCs) and generations.
    @param oeds: The Outlier Exposure dataset from which the OE subsets are sampled.
    @param logger: Some logger.
    """
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = []
    for i, ind in [(i, ind) for i, ind in enumerate(offspring) if ind in invalid_ind]:
        logger.print(f'Evaluate ind{i:03}..')
        fitnesses.append(toolbox.evaluate(ind))
        name = f'gen{gen:03}_ind{i:03}_fit{fitnesses[-1] * 100:06.3f}'
        logger.logimg(pt.join('individuals', name), torch.stack([oeds[id][0] for id in ind]), nrow=16)
        logger.logtxt(f'{name} with ids {ind}')
        ind.fitness.values = [fitnesses[-1]]
        node = tree.get(ind)
        node.content.file = pt.join(logger.dir, 'individuals', f"{name}.png")
        node.content.fitness = fitnesses[-1]

    pop[:] = offspring
    fits = [ind.fitness.values[0] for ind in pop]
    mean, std, minf, maxf = np.mean(fits), np.std(fits), np.min(fits), np.max(fits)
    history['pop'].append(pop)
    history['fit'].append(fits)
    history['mean_fit'].append(mean)
    history['std_fit'].append(std)
    history['min_fit'].append(minf)
    history['max_fit'].append(maxf)
    name = f'gen{gen:03}'
    logger.logimg(
        pt.join('raw_gen', name), torch.cat([torch.stack([oeds[id][0] for id in ind]) for ind in pop]), nrow=len(pop[0]),
    )
    tolog = sorted(list(zip(fits, torch.cat([torch.stack([oeds[id][0] for id in ind]) for ind in pop]))), key=lambda x: x[0])
    tolog_fits, tolog_imgs = list(zip(*tolog))
    logger.logimg(
        name, torch.stack(tolog_imgs), nrow=len(pop[0]),
        rowheaders=[f'{f*100:06.3f}' for f in tolog_fits]
    )

    logger.print(f'GENERATION {gen:03}')
    logger.print(f"  Min {minf*100:06.3f}")
    logger.print(f"  Max {maxf*100:06.3f}")
    logger.print(f"  Avg {mean*100:06.3f}")
    logger.print(f"  Std {std*100:06.3f}")
    logger.add_scalar('avg_fit', mean*100, gen, tbonly=False)
    logger.add_scalar('max_fit', maxf*100, gen, tbonly=False)
    tree.save(pt.join(logger.dir, 'evolution'))


def evolve(pop: List[object], gen: int, toolbox: base.Toolbox, mate_chance: float, mutation_chance: float,
           history: dict, tree: Tree, oeds: Subset, logger: Logger, log_mutations: bool = True):
    """
    Executes one step of the evolutionary algorithm. That is, takes the latest generation, selects
    survivors based on the tournament rule, mates them, mutates them, and finally evaluates all OE subsets of the
    resulting population. The results are logged in the history and genealogical tree.

    @param pop: The popoluation that is to be evolved (i.e., the latest generation).
    @param gen: The generation number; e.g., 0 for the first.
    @param toolbox: The prepared DEAP toolbox that provides functions for mating, etc.
    @param mate_chance: The chance to mate an OE subset with another one.
    @param mutation_chance: The chance to mutate an OE subset.
    @param history: The history dictionary that stores information about the setup and all evaluated generations.
    @param tree: The genealogical tree with all the fitness values (i.e., mean AUCs) and generations.
    @param oeds: The Outlier Exposure dataset from which the OE subsets are sampled.
    @param logger: Some logger.
    @param log_mutations: Whether to visualize and log the mutations.
    """
    logger.print('-------------------------------------------------------')
    logger.print(f'-------------------GENERATION {gen:03}----------------------')
    logger.print('-------------------------------------------------------')

    # ------------------------------- SELECT --------------------------------
    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))

    if log_mutations:
        old_samples = torch.cat([torch.stack([oeds[id][0] for id in ind]) for ind in pop])
        new_samples = torch.cat([torch.stack([oeds[id][0] for id in ind]) for ind in offspring])
        survivors = [i for i, ind in enumerate(pop) if ind in offspring]
        if len(pop[0]) > 1:
            nrow = len(pop[0])
            row_sep_at = (16, len(pop))
            mark = [j for i in survivors for j in list(range(i * len(pop[0]), (i + 1) * len(pop[0])))]
        else:
            nrow = len(pop)
            row_sep_at = (16, 1)
            mark = [survivors]
        logger.logimg(
            pt.join('selection', f'gen{gen:03}'), torch.cat([old_samples, new_samples]),
            nrow=nrow, row_sep_at=row_sep_at, mark=mark
        )
    # ----------------------------------------------------------------------

    # ------------------------------- MATE ---------------------------------
    before_mating, picked = list(map(toolbox.clone, offspring)), []
    for i, (child1, child2) in enumerate(zip(offspring[::2], offspring[1::2])):
        if random.random() < mate_chance:
            node1, node2 = tree.get(child1), tree.get(child2)
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
            picked.append(i)
            child1_node, child2_node = EvolNode(Individual(list(child1))), EvolNode(Individual(list(child2)))
            node1.add_children(child1_node, child2_node)
            if node1 != node2:
                node2.add_children(child1_node, child2_node)

    if log_mutations:
        old_samples = torch.cat([torch.stack([oeds[id][0] for id in ind]) for ind in before_mating])
        new_samples = torch.cat([torch.stack([oeds[id][0] for id in ind]) for ind in offspring])
        mated = [[p*2, p*2+1, len(pop) + p*2,  len(pop) + p*2+1] for p in picked]
        if len(pop[0]) > 1:
            nrow = len(pop[0])
            row_sep_at = (16, len(pop))
            mark = [[j for i in (a, b, c, d) for j in range(i * len(pop[0]), (i + 1) * len(pop[0]))] for a, b, c, d in mated]
        else:
            nrow = len(pop)
            row_sep_at = (16, 1)
            mark = mated
        logger.logimg(
            pt.join('mating', f'gen{gen:03}'), torch.cat([old_samples, new_samples]),
            nrow=nrow, row_sep_at=row_sep_at, mark=mark
        )
    # ----------------------------------------------------------------------

    # ------------------------------- MUTATE -------------------------------
    before_mutating, picked = list(map(toolbox.clone, offspring)), []
    for i, mutant in enumerate(offspring):
        if random.random() < mutation_chance:
            node = tree.get(mutant)
            toolbox.mutate(mutant)
            del mutant.fitness.values
            picked.append(i)
            child_node = EvolNode(Individual(list(mutant)))
            node.add_children(child_node)

    if log_mutations:
        old_samples = torch.cat([torch.stack([oeds[id][0] for id in ind]) for ind in before_mutating])
        new_samples = torch.cat([torch.stack([oeds[id][0] for id in ind]) for ind in offspring])
        mutated = [[p, len(pop) + p] for p in picked]
        if len(pop[0]) > 1:
            nrow = len(pop[0])
            row_sep_at = (16, len(pop))
            mark = [[j for i in (a, b) for j in range(i * len(pop[0]), (i + 1) * len(pop[0]))] for a, b in mutated]
        else:
            nrow = len(pop)
            row_sep_at = (16, 1)
            mark = mutated
        logger.logimg(
            pt.join('mutation', f'gen{gen:03}'), torch.cat([old_samples, new_samples]),
            nrow=nrow, row_sep_at=row_sep_at, mark=mark
        )
    # ----------------------------------------------------------------------

    evaluate(offspring, pop, gen, toolbox, history, tree, oeds, logger)

