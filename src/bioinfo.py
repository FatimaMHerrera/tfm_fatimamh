import warnings
import requests
import json
from Bio import BiopythonWarning
from Bio.Align import PairwiseAligner
warnings.simplefilter('ignore', BiopythonWarning)
from typing import Union, List, Tuple, Optional

class UnitProtInfo:

    __slots__ = ['unitprot', 'peptide']

    def __init__(self, unitprot, peptide):
        self.unitprot = unitprot
        self.peptide = peptide

    def obtener_secuencia_proteina(self):
        url = f"https://rest.uniprot.org/uniprotkb/{self.unitprot}.json"
        response = requests.get(url)

        if response.ok:
            data = response.json()
            return data.get('sequence', {}).get('value', '')
        else:
            print("Error en la solicitud:", response.status_code)
            return None

    def buscar_peptido_en_proteina(self):
        secuencia_proteina = self.obtener_secuencia_proteina()
        if secuencia_proteina:
            posicion = secuencia_proteina.find(self.peptide)
            if posicion != -1:
                return f"El péptido '{self.peptide}' se encuentra en la posición {posicion} de la proteína."
            else:
                return "El péptido no se encontró en la secuencia de la proteína."
        else:
            return "No se pudo obtener la secuencia de la proteína."
        
    def obtener_nombre_proteina_uniprot(self):
        base_url = f"https://rest.uniprot.org/uniprotkb/{self.unitprot}.json"
        response = requests.get(base_url)

        if response.ok:
            data = response.json()
            # Acceder al nombre recomendado de la proteína
            protein_name = data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'Nombre no encontrado')
            return protein_name
        else:
            print("Error en la solicitud:", response.status_code)
            return None
        
    def obtener_descripcion_go(self, go_id):
        url = f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{go_id}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and len(data['results']) > 0:
                return data['results'][0]['name']  # Retorna la descripción del término GO
            else:
                return "Descripción no disponible"
        else:
            return "Error al acceder a QuickGO para la descripción"

    def obtener_terminos_go_api(self):
        url = f"https://www.ebi.ac.uk/QuickGO/services/annotation/search?geneProductId={self.unitprot}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            terminos_go = {}
            for anotacion in data['results']:
                go_id = anotacion['goId']
                if go_id not in terminos_go:
                    terminos_go[go_id] = self.obtener_descripcion_go(go_id)
            return terminos_go
        else:
            return {"Error al acceder a QuickGO para anotaciones": []}
        
    
class SimilarityOfPeptides:
    """This class pretends to calculate (or give information) about the similarity between peptides."""

    __slots__ = ['peptide1', 'peptide2']

    def __init__(self, list_of_peptides: Optional[Union[List[str], Tuple[str]]]) -> None:
        self.peptide1 = list_of_peptides[0]
        self.peptide2 = list_of_peptides[1]

    def comparar_secuencias(self, type='local'):
        # Crear una instancia de PairwiseAligner
        aligner = PairwiseAligner()

        # Configurar el tipo de alineamiento
        if type == 'local':
            aligner.mode = 'local'
        elif type == 'global':
            aligner.mode = 'global'

        # Realizar el alineamiento
        alineamientos = aligner.align(self.peptide1, self.peptide2)

        # Imprimir los alineamientos
        for alineamiento in alineamientos:
            print(alineamiento)
