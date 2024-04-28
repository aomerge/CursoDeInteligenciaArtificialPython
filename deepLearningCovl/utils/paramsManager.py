import json 
class ParamsManager(object):
    """
    Clase que se encarga de almacenar y cargar los parámetros de un modelo
    """
    def __init__(self, path):
        """
        @param path: str: Ruta del archivo de parámetros
        """
        self.path = path
        self.params = json.load(open(self.path, 'r'))

    def getParams(self):
        """
        @return dict: Diccionario con los parámetros
        """
        return self.params
    
    def getAgentParams(self):
        """
        @param params: dict: Diccionario con los parámetros
        """
        return self.params['agent']

    def getEnvParams(self):
        """
        @return dict: Diccionario con los parámetros del entorno
        """
        return self.params['enviroment']
    
    def updateAgentParams(self, **kwargs):
        """
        @param agentParams: dict: Diccionario con los parámetros del agente
        """

        for key, value in kwargs.items():
            if key in self.getAgentParams().keys():
                self.params['agent'][key] = value
    
    def exportAgentParams(self, filename):
        """
        Exporta los parámetros del agente a un archivo
        """
        with open(filename, 'w') as f:
            json.dump(self.params['agent'], f, indent=4, separators=(',', ': '), sort_keys=True)
            f.write('\n')
    def exportEnvParams(self, filename):
        """
        Exporta los parámetros del entorno a un archivo
        """
        with open(filename, 'w') as f:
            json.dump(self.params['enviroment'], f, indent=4, separators=(',', ': '), sort_keys=True)
            f.write('\n')

if __name__ == '__main__':
    print("Testing ParamsManager")
    path = '../parameters.json'
    manager = ParamsManager(path) 
    agentParams = manager.getAgentParams()
    print("the parameters fo agent",agentParams)
    for key, value in agentParams.items():
        print(key," : " ,value)
    envParams = manager.getEnvParams()
    print("the parameters fo env",envParams)
    for key, value in envParams.items():
        print(key," : " ,value)