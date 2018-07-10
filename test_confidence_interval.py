from caserec.evaluation.rating_prediction import RatingPredictionEvaluation
from scipy.cluster.hierarchy import ward, to_tree
from scipy.spatial.distance import pdist

from confidence_interval import mean_confidence_interval
from caserec.utils.process_data import ReadFile, WriteFile
import numpy as np


train_file = '/home/fortesarthur/Documentos/fernando_paper/folds5/0/train.dat'
test_file = '/home/fortesarthur/Documentos/fernando_paper/folds5/0/test.dat'
prediction_file = '/home/fortesarthur/Documentos/fernando_paper/folds5/0/pred_wards.dat'

# # There are items with only one interaction: Warming
# for item in td['items']:
#     sample = []
#     for user in td['users_viewed_item'][item]:
#         sample.append(td['feedback'][user][item])
#     print(item, sample, mean_confidence_interval(sample))


class Alg(object):
    def __init__(self, tr, te, output_file=None, sep='\t', distance='cosine'):
        self.distance = distance
        self.output_file = output_file
        self.sep = sep
        self.train_set = ReadFile(tr).read()
        self.test_set = ReadFile(te).read()

        self.items = self.train_set['items']
        self.users = self.train_set['users']

        self.item_to_item_id = {}
        self.item_id_to_item = {}
        self.user_to_user_id = {}
        self.user_id_to_user = {}
        self.user_confidence = {}

        self.matrix = None
        self.father_of = {}
        self.leaves = []
        self.items_cluster = {}
        self.cluster_item_interval = {}
        self.predictions = []
        self.evaluation_results = {}

    def initialize(self):
        # map users and items
        for i, item in enumerate(self.items):
            self.item_to_item_id.update({item: i})
            self.item_id_to_item.update({i: item})

        # calculate confidence interval
        for u, user in enumerate(self.users):
            self.user_to_user_id.update({user: u})
            self.user_id_to_user.update({u: user})
            self.user_confidence[user] = mean_confidence_interval(list(self.train_set['feedback'][user].values()))

    def create_matrix(self):
        """
        Method to create a feedback matrix
        """

        self.matrix = np.zeros((len(self.users), len(self.items)))

        for user in self.train_set['users']:
            for item in self.train_set['feedback'][user]:
                self.matrix[self.user_to_user_id[user]][self.item_to_item_id[item]] = \
                    self.train_set['feedback'][user][item]

    def compute_distance(self, transpose=False):
        """
        Method to compute a similarity matrix from original df_matrix
        """

        # Calculate distance matrix
        if transpose:
            distance_matrix = pdist(self.matrix.T, self.distance)
        else:
            distance_matrix = pdist(self.matrix, self.distance)

        # Remove NaNs
        distance_matrix[np.isnan(distance_matrix)] = 1.0

        return distance_matrix

    def create_hierarchy(self):
        hr = ward(self.compute_distance())
        root, hr_tree = to_tree(hr, True)
        self.return_father_pointers(hr_tree, root)
        self.find_items_in_clusters(hr_tree)

    def return_father_pointers(self, hr_tree, root):
        self.father_of[root.id] = None

        for c in range(len(hr_tree)):
            if hr_tree[c].is_leaf():
                self.leaves.append(c)
            else:
                self.father_of[hr_tree[c].left.id] = c
                self.father_of[hr_tree[c].right.id] = c

    def find_items_in_clusters(self, hr_tree):
        splits = dict()

        for solutionCluster in range(len(hr_tree)):
            pre_order_visit = hr_tree[solutionCluster].pre_order()

            # getting all clusters below this one (in pre_order)
            splits[solutionCluster] = []
            # for each cluster verify if it is a sub-cluster or a single object, if is a
            # single object (leaf) adds to the output
            # This way the output will have the clusterID mapping to it's objects
            for c in pre_order_visit:
                if hr_tree[c].is_leaf():
                    splits[solutionCluster].append(self.user_id_to_user[c])

            # getting all items of each cluster
            if len(splits[solutionCluster]) > 1:
                for user in splits[solutionCluster]:
                    self.items_cluster.setdefault(solutionCluster,
                                                  set()).update(self.train_set['items_seen_by_user'][user])

        for cluster in self.items_cluster:
            for item in self.items_cluster[cluster]:
                self.cluster_item_interval.setdefault(cluster, {}).update({item: []})
                for user in splits[cluster]:
                    if self.train_set['feedback'][user].get(item, []):
                        self.cluster_item_interval[cluster][item].append(self.train_set['feedback'][user][item])

                self.cluster_item_interval[cluster][item] = \
                    mean_confidence_interval(self.cluster_item_interval[cluster][item], confidence=.95)

        # item = 56
        # cluster = hr_tree[len(hr_tree) - 1]
        # while cluster is not None:
        #     if not cluster.is_leaf():
        #         if self.cluster_item_interval[cluster.id].get(item, []):
        #             print(self.cluster_item_interval[cluster.id][item][1])
        #         else:
        #             print('Tem porra nenhuma!')
        #     else:
        #         print("Leave: SaPorra! ")
        #     old_cluster = cluster
        #     cluster = old_cluster.get_right()

    def evaluate(self, metrics, verbose=True, as_table=False, table_sep='\t'):
        """
        Method to evaluate the final ranking

        :param metrics: List of evaluation metrics
        :type metrics: list, default ('MAE', 'RMSE')

        :param verbose: Print the evaluation results
        :type verbose: bool, default True

        :param as_table: Print the evaluation results as table
        :type as_table: bool, default False

        :param table_sep: Delimiter for print results (only work with verbose=True and as_table=True)
        :type table_sep: str, default '\t'

        """

        if metrics is None:
            metrics = list(['MAE', 'RMSE'])

        results = RatingPredictionEvaluation(verbose=verbose, as_table=as_table, table_sep=table_sep, metrics=metrics
                                             ).evaluate_recommender(predictions=self.predictions,
                                                                    test_set=self.test_set)

        for metric in metrics:
            self.evaluation_results[metric.upper()] = results[metric.upper()]

    def recommendation_step(self):
        for user in self.test_set['users']:
            user_id = self.user_to_user_id[user]
            bu, hu = mean_confidence_interval(list(self.train_set['feedback'][user].values()), confidence=.95)

            for item in self.test_set['items_seen_by_user'][user]:
                cluster = self.father_of[user_id]

                '''
                mi^k - > media das notas do item em um subconjunto k
                mu -> media das notas do user u 
                * utilizar o h -> a diferenca entra a media e a borda do intervalo (Soh para subir a arvore?)
                
                rui = (wi * mi^k + wu * mu) / (wi + wu) 
                
                '''

                bi = 0
                last_h = float('inf')

                while True:
                    if cluster is None:
                        break

                    if self.cluster_item_interval[cluster].get(item, -1) == -1:
                        cluster = self.father_of[cluster]
                    else:
                        new_h = self.cluster_item_interval[cluster][item][1]

                        if np.isnan(new_h) or new_h == 0:
                            bi = self.cluster_item_interval[cluster][item][0]
                            cluster = self.father_of[cluster]

                        elif new_h < last_h:
                            last_h = new_h
                            bi = self.cluster_item_interval[cluster][item][0]
                            cluster = self.father_of[cluster]

                        else:
                            cluster = self.father_of[cluster]

                if bi == 0:
                    rui = bu
                else:
                    rui = .5 * bu + .5 * bi

                self.predictions.append((user, item, rui))

        self.predictions = sorted(self.predictions, key=lambda x: x[1])

        if self.output_file is not None:
            WriteFile(self.output_file, data=self.predictions, sep=self.sep).write()

    def compute(self):
        self.initialize()
        self.create_matrix()
        self.create_hierarchy()
        self.recommendation_step()

        self.evaluate(metrics=None)


Alg(train_file, test_file, prediction_file).compute()
