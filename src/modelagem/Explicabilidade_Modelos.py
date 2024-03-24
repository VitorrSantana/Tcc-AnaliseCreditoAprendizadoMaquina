import shap
from sklearn.model_selection import train_test_split

shap.initjs()

class Explicabilidade_Modelos:

    def __init__(self,model,type_model='tree') -> None:
        
        self.model = model
        if type_model == 'tree':
            self.explainer = shap.TreeExplainer(model)
        else: # mudar para adicionar mais modalidades
            pass

    # Passar dados que tenha marcado Treino Teste e Validação
    # Passar cada dado por vez
    def set_data(self,X,y,amosEst=True,tipo_dado='Treino',porc_avaliada = 0.3,):
        
        if amosEst:
            if tipo_dado =='Treino':
                self.X, _, _, _  = train_test_split(X,y, train_size=porc_avaliada, random_state=1)
            if tipo_dado =='Teste':
                _, self.X, _, _   = train_test_split(X,y, test_size=porc_avaliada,random_state=1)
        else:
            self.X                = X.sample(frac=porc_avaliada, random_state=1)
        
        self.shap_values      = self.explainer.shap_values(self.X) 
        
                
        
    def grafico_force_plot_instace_unique(self,col_identificacao = 'SK_ID_CURR',id_cliente=None):
        
        if id_cliente is None:
            idx_cliente = 0
        else:
            idx_cliente = self.X.query(f'{col_identificacao}=={id_cliente}').index
        
        shap.force_plot(self.explainer.expected_value[1],self.shap_values[1][idx_cliente,:],self.X.iloc[idx_cliente,:])


    def grafico_force_plot_instances(self):
        # Adicionar Regra para diferentes bases
        shap.force_plot(self.explainer.expected_value[1],self.shap_values[1],self.X)

    def grafico_summary_plot(self):
         
        shap.summary_plot(self.shap_values[1],self.X)


    def grafico_dependence_plot(self,feature_dependence,intr_index = None):

        shap.dependence_plot(self.shap_values[1],self.X,interaction_index=intr_index)

        