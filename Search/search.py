import os
import random
import sys

import torch
import yaml
from tqdm import tqdm
from Search.evaluate import evaluate
from Model.model import HasNet
from Utils.utils import *
from Utils.Parser import Parser

from Utils.dataLoader import *

logger = log1('test', 'search')

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

choice = lambda x: x[np.random.randint(len(x))] if isinstance(x, tuple) else choice(tuple(x))


class EvolutionSearcher(object):
    def __init__(self, config, initialArch, validData):
        self.maxEpochs = config['maxEpochs']
        self.k = config['topSampleNum']
        self.population = config['population']
        self.mProb = config['mProb']
        self.crossoverNum = config['crossoverNum']
        self.mutationNum = config['mutationNum']

        self.latencyLimit = config['latencyLimit']
        self.exp_no = config['exp_no']
        self.reverse = config['reverse']
        self.dataLoader = validData

        self.model = HasNet()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device == 'cuda':
            self.model.cuda()
        self.model.load_state_dict(torch.load(f'../weights/{self.exp_no}/supernet/checkpoint-latest.pth', map_location=torch.device('cpu')))

        self.initialArch = initialArch
        self.layerNum = 8
        self.choiceNum = 8
        self.candidates_log = []
        self.visitDict = {}
        self.topKDict = {self.k: []}
        self.epoch = 0
        self.candidates = []

    def initialization(self):
        torch.manual_seed(0)
        if self.device == 'cuda':
            torch.cuda.set_device(0)
            torch.backends.cudnn.deterministic = True
        np.random.seed(0)
        random.seed(0)
        sys.setrecursionlimit(10000)
        if not self.loadCheckpoint():
            self.getRandom(self.population)

    def legitimacyJudge(self, cand):
        """
        Judge the latency, initial the parameter of the cand
        :param cand: generated architecture
        :return:
        """
        assert isinstance(cand, tuple) and len(cand) == self.layerNum
        if cand not in self.visitDict:
            self.visitDict[cand] = {}
        info = self.visitDict[cand]
        if 'visited' in info:
            return False

        if 'latency' not in info:
            info['latency'] = getLatency(cand)
        if info['latency'] > self.latencyLimit:
            log('latency limit exceed')
            logger.info('latency limit exceed')
            return False

        info['score'] = evaluate(self.model, cand, self.dataLoader, self.device)
        info['visited'] = True

        return True

    def updateTopK(self, candidates, k, key, reverse=False):
        assert k in self.topKDict
        log('select......')
        logger.info('select......')
        t = self.topKDict[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.topKDict[k] = t[:k]
        print(self.topKDict)

    def stackRandomCand(self, randomFunc, batch_size=10):
        """
        :return: iterator of candidates
        """
        while True:
            cands = [randomFunc() for _ in range(batch_size)]
            for cand in cands:
                if cand not in self.visitDict:
                    self.visitDict[cand] = {}
            for cand in cands:
                yield cand

    def getRandom(self, num):
        """
        Randomly generate the individuals, updated self.candidates
        :param num: total group number
        :return: None
        """
        log('random select ......')
        logger.info('random select ......')
        candIter = self.stackRandomCand(
            lambda: tuple(np.random.randint(self.choiceNum) for i in range(self.layerNum)))
        # while len(self.candidates) < num:
        #     cand = next(candIter)
        #     if not self.legitimacyJudge(cand):
        #         continue
        #     self.candidates.append(cand)
        #     log(f'random {len(self.candidates)}/{num}')
        # log('random_num = {}'.format(len(self.candidates)))
        if len(self.candidates) < num:
            for i in tqdm(range(num - len(self.candidates))):
                cand = next(candIter)
                if not self.legitimacyJudge(cand):
                    continue
                self.candidates.append(cand)
        log(f'random_num = {len(self.candidates)}')
        logger.info(f'random_num = {len(self.candidates)}')


    def Mutation(self, k, mutationNum, mProb):
        assert k in self.topKDict
        log('mutation......')
        logger.info('mutation......')
        out = []
        max_iters = mutationNum * 10

        candIter = self.stackRandomCand(
            lambda: tuple(np.random.randint(self.choiceNum) for i in range(self.layerNum)))
        if max_iters > 0:
            for i in tqdm(mutationNum):
                max_iters -= 1
                cand = next(candIter)
                if not self.legitimacyJudge(cand):
                    continue
                out.append(cand)

        log(f'mutationNum = {len(out)}')
        logger.info(f'mutationNum = {len(out)}')
        return out

    def Crossover(self, k, crossoverNum):
        assert k in self.topKDict
        log('crossover......')
        logger.info('crossover......')
        out = []
        maxIter = crossoverNum*10

        candIter = self.stackRandomCand(
            lambda: tuple(np.random.randint(self.choiceNum) for i in range(self.layerNum)))
        if maxIter > 0:
            for i in tqdm(range(crossoverNum)):
                maxIter -= 1
                cand = next(candIter)
                if not self.legitimacyJudge(cand):
                    continue
                out.append(cand)

        log(f'crossoverNum = {len(out)}')
        logger.info(f'crossoverNum = {len(out)}')
        return out

    def saveCheckpoint(self):
        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')

        info = {'candidates_log': self.candidates_log,
                'candidates': self.candidates,
                'visitDict': self.visitDict,
                'topKDict': self.topKDict,
                'epoch': self.epoch}

        if not os.path.exists(f'../weights/{self.exp_no}/search'):
            os.mkdir(f'../weights/{self.exp_no}/search')
        checkpointPath = f'../weights/{self.exp_no}/search/checkpoint.pth.tar'
        torch.save(info, checkpointPath)
        log(f'save checkpoint to {checkpointPath}')
        logger.info(f'save checkpoint to {checkpointPath}')

    def loadCheckpoint(self):
        if not self.exp_no == 0:
            checkpointPath = f'../weights/{self.exp_no - 1}/search/checkpoint.pth.tar'
        else:
            return False

        info = torch.load(checkpointPath)
        self.candidates_log = info['candidates_log']
        self.candidates = info['candidates']
        self.visitDict = info['visitDict']
        self.topKDict = info['topKDict']
        self.epoch = info['epoch']

    def search(self):
        log(f'population = {self.population}, topSample = {self.k}, mutation = {self.mutationNum},'
            f'crossover = {self.crossoverNum}, maxEpochs = {self.maxEpochs}')
        logger.info(f'population = {self.population}, topSample = {self.k}, mutation = {self.mutationNum},'
            f'crossover = {self.crossoverNum}, maxEpochs = {self.maxEpochs}')

        self.initialization()
        self.updateTopK(self.candidates, k=self.k,
                        key=lambda x: self.visitDict[x]['score'], reverse=self.reverse)

        while self.epoch < self.maxEpochs:
            mutation = self.Mutation(self.k, self.mutationNum, self.mProb)
            crossover = self.Crossover(self.k, self.crossoverNum)
            self.candidates = mutation + crossover

            self.getRandom(self.population)
            self.updateTopK(self.candidates, k=self.k,
                            key=lambda x: self.visitDict[x]['score'], reverse=self.reverse)

            self.epoch += 1
            self.candidates_log.append(self.topKDict[self.k])
            log(f'epoch = {self.epoch}, top{len(self.topKDict[self.k])} result')
            logger.info(f'epoch = {self.epoch}, top{len(self.topKDict[self.k])} result')
            for i, cand in enumerate(self.topKDict[self.k]):
                ops = [i for i in cand]
                print(f'No. {i + 1},{cand} score = {self.visitDict[cand]["score"]}, ops = {ops}')
                logger.info(f'No. {i + 1},{cand} score = {self.visitDict[cand]["score"]}, ops = {ops}')

        self.saveCheckpoint()


if __name__ == "__main__":
    parser = Parser()
    args = parser.parse()

    if args.checkpoint:
        pretrain = torch.load
    args.config = "./config.yaml"

    with open(args.config) as f:
        config = yaml.load(f)

    model = HasNet()
    if args.checkpoint:
        paramSupernet = torch.load(f'../weights/{config["exp_no"]}/supernet/checkpoint-latest.pth', map_location=torch.device('cpu'))
        model.load_state_dict(paramSupernet)
        log('Supernet load finished.')
        logger.info('Supernet load finished')

    vailPath = '/Users/luoyongjia/Research/Data/MIT/validation'
    # vailPath = '/root/data/lyj/data/MIT5K/validation'
    vailListPath = buildDatasetListTxt(vailPath)

    vailData = loadDataset(vailPath, vailListPath, cropSize=config['length'], toRAM=True, training=False)
    vailLoader = DataLoader(vailData, batch_size=config['batchSize'])

    t = time.time()
    searcher = EvolutionSearcher(config, tuple([1] * 8), vailLoader)
    searcher.search()

    print(f'total searching time: {(time.time() - t) / 3600 :.2f} hours')
    logger.info(f'total searching time: {(time.time() - t) / 3600 :.2f} hours')


