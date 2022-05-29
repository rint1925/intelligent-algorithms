import sys
import numpy as np


REPEAT_NUM = 1000 # 繰り返し数
ANT_NUM = 100 # アリの数
PHERO_Q = 10 # 1回の巡回で分泌するフェロモン量
EVA_R = 0.05 # フェロモンの蒸発率
PHERO_R = 0.95 # フェロモンに基づいて経路を選択する確率
PHERO_L = 1 # フェロモンを考慮する度合い
HEU_L = 1 # ヒューリスティック情報を考慮する度合い
RAND_01 = np.random.rand() # 0以上1以下の実数乱数


# コロニーのクラス
class Colony():
    
    def __init__(self, file_name):
        self.field = Field(file_name) # 採餌行動の場
        self.ant = [Ant(arg_colony = self) for i in range(ANT_NUM)] # コロニーのメンバー(Antオブジェクトが格納される配列)
        self.nume = np.zeros((self.field.node_num, self.field.node_num)) # iノードにいるアリがj番ノードに進む確率の分子 nume[i][j] 



    # 経路を選択する
    def select_route(self):
        
        # 確率の分子を算出する
        for i in range(self.field.node_num):
            
            j = 1
            while(j < i):
                self.nume[i][j] = (self.field.pheromone[j][i] ** PHERO_L) * ((1 / self.field.distance[i][j]) ** HEU_L)
                j += 1

            j = i + 1
            while(j < self.field.node_num):
                self.nume[i][j] = (self.field.pheromone[i][j] ** PHERO_L) * ((1 / self.field.distance[i][j]) ** HEU_L)
                j += 1
    
        # 経路を選択する
        for i in range(ANT_NUM):
            self.ant[i].select_route()

        return None

    
    # フェロモン量を更新する
    def renew_pheromone(self):

        # 蒸発させる
        for i in range(self.field.node_num):
            j = i + 1
            while(j < self.field.node_num):
                self.field.pheromone[i][j] *= (1 - EVA_R)
                j += 1

        # アリによる追加分を加算する
        for i in range(ANT_NUM):
            self.ant[i].put_pheromone()

        return None

    
    # フェロモン量を表示する
    def print_pheromone(self):
        for i in range(self.field.node_num):
            for j in range(self.field.node_num):
                print('{:>9.3f}'.format(self.field.pheromone[i][j]), end=', ')
            print()
        
        return None




# アリのクラス
class Ant():

    def __init__(self, arg_colony:Colony):
        self.colony = arg_colony # 属しているコロニー
        self.route = np.zeros(self.colony.field.node_num, dtype=int) # 経路 (int)
        self.candidate = np.zeros(self.colony.field.node_num, dtype=int) # 未訪問ノード (i番のノードが選ばれているとき、candidate[i] == 0)
        self.total_dis = 0.0 # 現在の経路の総移動距離 


    # 経路を選択する
    def select_route(self) -> None:

        # 未訪問ノードを初期化する
        i = 1
        while(i < self.colony.field.node_num):
            self.candidate[i] = 1
            i += 1 

        # 経路を選択する
        self.total_dis = 0.0
        for i in range(self.colony.field.node_num -2):

            # 確率の分母を算出する
            denom = 0.0
            j = 1
            while(j < self.colony.field.node_num):
                if self.candidate[j] == 1:
                    denom += self.colony.nume[self.route[i]][j]
                j += 1
            
            # 次のノードを選択する
            next_node = -1
            if (denom != 0) and (RAND_01 <= PHERO_R):
                
                # フェロモン量に基づいて選択する
                r = RAND_01
                next_node = 1
                while(next_node < self.colony.field.node_num):
                    if self.candidate[next_node] == 1:
                        prob = self.colony.nume[self.route[i]][next_node] / denom

                        if r <= prob:
                            break

                        r -= prob

                    next_node += 1

                if (next_node == self.colony.field.node_num):
                    next_node = -1


            if (next_node == -1):
                # ランダムに選択する
                next_node2 = np.random.randint(0, self.colony.field.node_num -i-1)
                next_node = 1
                while (next_node < self.colony.field.node_num - 1):
                    if (self.candidate[next_node] == 1):
                        if (next_node2 == 0):
                            break
                        else:
                            next_node2 -= 1

                    next_node += 1  

            self.route[i+1] = next_node
            self.candidate[next_node] = 0
            self.total_dis += self.colony.field.distance[self.route[i]][next_node]



        # 最後の1ノードを探索する
        next_node = 1
        while (next_node < self.colony.field.node_num):
            if self.candidate[next_node] == 1:
                break

            next_node += 1

        self.route[self.colony.field.node_num - 1] = next_node
        self.total_dis += self.colony.field.distance[self.route[self.colony.field.node_num - 2]][next_node]


        # 出発地点への距離を加算する
        self.total_dis += self.colony.field.distance[next_node][0]

        return None

    
    # フェロモンを分泌する
    def put_pheromone(self):
        p = PHERO_Q / self.total_dis
        for i in range(self.colony.field.node_num - 1):
            if(self.route[i] < self.route[i+1]):
                self.colony.field.pheromone[self.route[i]][self.route[i+1]] += p
            else:
                self.colony.field.pheromone[self.route[i+1]][self.route[i]] += p

        self.colony.field.pheromone[0][self.route[self.colony.field.node_num - 1]] += p

        return None





# 土地のクラス
class Field():

    def __init__(self, file_name:str):

        data = np.loadtxt(file_name, delimiter=',')
        if data.shape[0] != data.shape[1]:
            print('列数と行数が一致しません')
            sys.exit(1)

        self.node_num = data.shape[0] # ノード数 (int)
        self.distance = data # ノード間距離
        self.pheromone = np.zeros((self.node_num, self.node_num)) # エッジのフェロモン量. i<jである要素にだけ値を格納.



if __name__ == "__main__":
    colony = Colony("sampledata.csv")
    for i in range(REPEAT_NUM):
        colony.select_route()
        colony.renew_pheromone()
    
    colony.print_pheromone()