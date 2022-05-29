import numpy as np

# 個体群のクラス
class Population:

    # コンストラクタ
    def __init__(self, pop_size: int, elite_num :int,
                 mutate_prob: float, n: int) -> None:
    
        self.POP_SIZE = pop_size  # 個体群のサイズ
        self.ELITE = elite_num # エリート保存戦略で残す個体の数
        self.MUTATE_PROB = mutate_prob # 突然変異確率
        self.N = n # 集合の要素となる最大数の平方値
        
        # 現世代個体群と次世代個体群の初期化
        self.inds = []
        self.next_inds = []
        for i in range(self.POP_SIZE):
            self.inds.append(Indivisual(self.MUTATE_PROB, self.N))
            self.next_inds.append(Indivisual(self.MUTATE_PROB, self.N))

        for _, ind in enumerate(self.inds):
            ind.evaluate()
        self.__evaluate()

        for _, next_ind in enumerate(self.next_inds):
            next_ind.evaluate()

        return None


    # デストラクタ
    def __del__(self) -> None:
        
        for _, ind in enumerate(self.inds):
            del ind
        for _, next_ind in enumerate(self.next_inds):
            del next_ind
        
        del self.inds
        del self.next_inds
        
        return None

    # 世代交代をする
    def alternate(self) -> None:
        
        # エリート保存戦略で子個体を作る
        for i in range(self.ELITE):
            self.next_inds[i] = self.inds[i]

        # 親を選択し交叉する
        for i in range(self.ELITE, self.POP_SIZE):
            p1 = self.__select()
            p2 = self.__select()
            self.next_inds[i].crossover(self.inds[p1], self.inds[p2])

        # 突然変異を起こす
        for i in range(self.POP_SIZE):
            self.next_inds[i].mutate()
        
        # 次世代を現世代に変更する
        tmp = self.inds
        self.inds = self.next_inds
        self.next_inds = tmp

        # 評価する
        self.__evaluate()

        return None


    # 結果を表示する
    def print_result(self) -> None:
        print('\n集合A: ')
        for i in range(self.N):
            if (self.inds[0].chrom[i] == 1):
                print('√'+str(i+1))
        
        print('\n集合B: ')
        for i in range(self.N):
            if (self.inds[0].chrom[i] == 0):
                print('√'+str(i+1))

        print('\n差: '+str(self.inds[0].fitness))
        return None



    # すべての個体を評価して,適応度順に並べ替える
    def __evaluate(self) -> None:
        for _, ind in enumerate(self.inds):
            ind.evaluate()
        self.__sort()

        return None


    # 親個体を1つ選択する
    def __select(self) -> int:
        # ルーレット選択
        # 戻り値は選択した親個体の添え字

        trf = [None] * self.POP_SIZE  # 適応度を変換(スケーリング)した値 
        denom = 0.0  # ルーレット選択の確率を決める時の分母(denominator)
        for i in range(self.POP_SIZE):
            trf[i] = (self.inds[-1].fitness - self.inds[i].fitness)\
                    /(self.inds[-1].fitness - self.inds[0].fitness)
            denom += trf[i]

        r = np.random.random() # ルーレット値 (0~1の実数)
        for rank in range(self.POP_SIZE):
            prob = trf[rank] / denom
            if(r <= prob):
                break
            r -= prob

        return rank


    # 個体を良い順に並び替える
    def __sort(self) -> None:
        self.inds = sorted(self.inds,key=lambda ind: ind.fitness)
        return None




# 個体のクラス
class Indivisual():

    # コンストラクタ
    def __init__(self, mutate_prob: float, n: int) -> None:
        self.MUTATE_PROB = mutate_prob
        self.N = n

        self.chrom = [0] * self.N  # 染色体 (ランダムで1か0が入っている)
        for i in range(self.N):
            self.chrom[i] = np.random.randint(0,2)
        self.fitness = 0.0 # 適応度
    
        return None


    # 適応度を算出する
    def evaluate(self) -> None:
        self.fitness = 0.0
        for i, bit in enumerate(self.chrom):
            self.fitness += (bit * 2 - 1) * np.sqrt(i + 1)
        self.fitness = np.abs(self.fitness)
        return None


    # 交叉による子にする
    def crossover(self, p1: 'Indivisual', p2: 'Indivisual') -> None:
        
        # p1とp2から一点交叉で作った子にする
        point = np.random.randint(0,self.N)

        for i in range(point):
            self.chrom[i] = p1.chrom[i]
        for i in range(point, self.N):
            self.chrom[i] = p2.chrom[i]

        return None


    # 突然変異を起こす
    def mutate(self) -> None:

        for i in range(self.N):
            if(np.random.random() < self.MUTATE_PROB):
                self.chrom[i] = 1 - self.chrom[i]  # 0と1を反転

        return None

    

# main関数
def main():

    GEN_MAX = 10
    POP_SIZE = 1000  # 個体群のサイズ
    ELITE = 1 # エリート保存戦略で残す個体の数
    MUTATE_PROB = 0.01 # 突然変異確率
    N = 64 # 集合の要素となる最大数の平方値

    pop = Population(pop_size=POP_SIZE, elite_num=ELITE,
                     mutate_prob=MUTATE_PROB, n=N)

    for i in range(GEN_MAX):
        pop.alternate()
        print('第' + str(i) + '世代 : 最良適応度 = ' + str(pop.inds[0].fitness))

    pop.print_result()
    del pop


if __name__ == "__main__":
    main()