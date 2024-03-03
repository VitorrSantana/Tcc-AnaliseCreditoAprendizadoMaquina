from src.pre_processamento.PreProcessamento import PreProcessamento
from src.modelagem.Modelagem                import Modelagem

import pandas   as pd
import warnings
import gc   

warnings.filterwarnings("ignore")

class Testes:

    def __init__(self,modelos:dict,data_final,colunas) -> None:
        self.modelos_p_Tete = modelos
        self.data_final     = data_final
        self.colunas        = colunas

    def executa_simulações(self,qtd_it_features=5,qtd_iteracoes=50,seletor_features=['person','mutal_info','anova']):
        resultado  = pd.DataFrame({'auc-roc':[],'params':[],'modelo':[],'feature_selector':[],'qtd_features':[]})
        idx_resultado = 0

        for seletor_feature in seletor_features:
            print(f'-------- Seletor Feature {seletor_feature} iniciando execução --------')
            pre_proce = PreProcessamento(self.data_final[self.colunas])
            match seletor_feature:
                case 'mutal_info': features = pre_proce.mutual_information(self.data_final[self.colunas],self.data_final['TARGET'])
                case 'person'    : features = pre_proce.correlacao_target(self.data_final[self.colunas],self.data_final['TARGET'])
                case 'anova'     : features = pre_proce.teste_f_classif(self.data_final[self.colunas],self.data_final['TARGET'])
            print(f'Colunas base {len(list(self.data_final[self.colunas].columns))}, Colunas selecionadas {len(features)}')
            for nome_modelo,parametros in  self.modelos_p_Tete.items():
                print(f'-------- Modelo {nome_modelo} iniciando execução --------')
                for idx_feature in range(0,len(features)-qtd_it_features-1,qtd_it_features):
                    features_selecionadas = features[0:qtd_it_features+idx_feature]
                    features_selecionadas += ['SK_ID_CURR','TARGET']

                    modelo  = Modelagem(self.data_final[features_selecionadas].drop('SK_ID_CURR',axis=1),'TARGET')
                    modelo.set_model(nome_modelo,{'random_state':42})

                    study,melhores_param,auc_roc = modelo.otimizacao_parametros_optuna(parametros,num_iteracoes=qtd_iteracoes)

                    resultado.loc[idx_resultado] = [auc_roc,melhores_param,nome_modelo,seletor_feature,qtd_it_features+idx_feature] 

                    del  modelo
                    gc.collect()
                    
                    idx_resultado +=1
        resultado.to_csv(f'resultados_teste_nome_modelo{nome_modelo}_{seletor_feature}.csv',index=False)
        print(resultado)




    