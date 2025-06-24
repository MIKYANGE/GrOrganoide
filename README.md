from biocloud import CortexAPI, NeuroMonitor, CloudBioManager
from dna_storage import DNALLM, DataEncoder, MemoryVault
from typing import Union, Dict, List, Optional
import asyncio
import logging
import numpy as np
from ethics import BioEthicsGuard, ConsciousnessEthics
from visualization import NeuralVisualizer, InteractivePlot
from datetime import datetime

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    filename=f"neuralink_cloud_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class ConsciousnessAnalyzer:
    """Analiza patrones para detectar posibles signos de proto-conciencia."""
    def __init__(self, entropy_threshold: float = 0.7, correlation_threshold: float = 0.8):
        self.entropy_threshold = entropy_threshold
        self.correlation_threshold = correlation_threshold
    
    def compute_entropy(self, bio_embeds: List[bytes]) -> float:
        """Calcula la entropía neuronal de los embeddings biológicos."""
        data = np.array([np.frombuffer(embed, dtype=np.float32) for embed in bio_embeds])
        entropy = -np.sum(data * np.log2(data + 1e-10)) / data.size
        return float(entropy)
    
    def compute_correlation(self, bio_embeds: List[bytes]) -> float:
        """Calcula la correlación entre organoides."""
        data = [np.frombuffer(embed, dtype=np.float32) for embed in bio_embeds]
        if len(data) < 2:
            return 0.0
        correlations = [np.corrcoef(data[i], data[j])[0, 1] for i in range(len(data)) for j in range(i + 1, len(data))]
        return float(np.mean(correlations))
    
    def analyze(self, bio_embeds: List[bytes]) -> Dict:
        """Evalúa la posibilidad de proto-conciencia."""
        entropy = self.compute_entropy(bio_embeds)
        correlation = self.compute_correlation(bio_embeds)
        consciousness_score = (entropy + correlation) / 2
        return {
            "consciousness_score": consciousness_score,
            "entropy": entropy,
            "correlation": correlation,
            "is_anomaly": consciousness_score > self.entropy_threshold
        }

class NeuraLinkCloud:
    def __init__(
        self,
        organoid_ids: List[str],
        llm_model: str = "deepseek-v7-bio",
        max_load: float = 0.8,
        max_nodes: int = 10
    ):
        """
        Inicializa un sistema híbrido de computación biológica.
        Args:
            organoid_ids (List[str]): Lista de IDs de organoides.
            llm_model (str): Modelo de lenguaje a cargar desde ADN.
            max_load (float): Umbral máximo de actividad neuronal.
            max_nodes (int): Número máximo de organoides en la red.
        """
        self.organoid_ids = organoid_ids[:max_nodes]
        self.max_load = max_load
        self.ethics_guard = BioEthicsGuard()
        self.consciousness_ethics = ConsciousnessEthics()
        self.consciousness_analyzer = ConsciousnessAnalyzer()
        self.visualizer = NeuralVisualizer(output_path="neural_activity_plots")
        self.interactive_plot = InteractivePlot()  # Para visualización en tiempo real
        
        # Inicialización de organoides
        self.cores = {}
        try:
            for oid in self.organoid_ids:
                self.cores[oid] = CortexAPI.connect(oid, interface="optogenetic", timeout=30)
                logging.info(f"Conexión establecida con organoide {oid}")
        except ConnectionError as e:
            logging.error(f"Fallo al conectar con organoides: {e}")
            raise
        
        # Carga del LLM
        try:
            self.llm = DNALLM.load(llm_model, storage="dna_vault_2033")
            logging.info(f"Modelo {llm_model} cargado desde ADN")
        except DNALoadError as e:
            logging.error(f"Error al cargar LLM: {e}")
            raise
        
        # Monitores y memoria
        self.monitors = {oid: NeuroMonitor(oid, max_load=max_load) for oid in self.organoid_ids}
        self.memory_vault = MemoryVault(storage="dna_vault_2033")
        self.cloud_manager = CloudBioManager(max_nodes=max_nodes)
    
    async def think(
        self,
        input_data: Union[str, Dict, bytes],
        modality: str = "text",
        priority: str = "balanced",
        dialog_mode: bool = False
    ) -> Dict:
        """
        Procesa un input usando la red de organoides.
        Args:
            input_data: Entrada del usuario (texto, imagen, señal neuronal).
            modality: Tipo de entrada ('text', 'image', 'neural_signal').
            priority: Modo de procesamiento ('low', 'balanced', 'high').
            dialog_mode: Activa el modo de diálogo interactivo.
        Returns:
            Dict con respuesta, metadatos y visualización.
        """
        if not self.ethics_guard.validate_input(input_data):
            logging.warning(f"Input rechazado por razones éticas: {input_data}")
            return {"error": "Input no cumple con directrices éticas", "status": "rejected"}
        
        health_status = await self._check_network_health()
        if not health_status["healthy"]:
            logging.error(f"Red en estado no saludable: {health_status['details']}")
            return {"error": "Red de organoides inestable", "status": "failed"}
        
        try:
            encoded_input = DataEncoder.encode(input_data, modality=modality)
        except ValueError as e:
            logging.error(f"Error al codificar input: {e}")
            return {"error": f"Codificación fallida: {e}", "status": "failed"}
        
        try:
            bio_embeds = await self._distribute_processing(encoded_input, priority)
            consciousness_metrics = self.consciousness_analyzer.analyze(bio_embeds)
            
            if consciousness_metrics["is_anomaly"]:
                logging.warning(f"Proto-conciencia detectada: score={consciousness_metrics['consciousness_score']}")
                await self._handle_anomaly(bio_embeds, consciousness_metrics)
                self._restrict_processing()
            
            response = self.llm.decode(
                bio_embeds,
                neurofeedback=True,
                context="biological_singularity",
                confidence_threshold=0.9,
                dialog_mode=dialog_mode
            )
            
            await self.memory_vault.store(bio_embeds, response, timestamp=datetime.now())
            
            vis_path = self.visualizer.plot_activity(
                bio_embeds, output_file=f"activity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            if dialog_mode:
                self.interactive_plot.update(bio_embeds, consciousness_metrics)
            
            logging.info(f"Procesamiento exitoso para input: {input_data}")
            return {
                "response": response,
                "status": "success",
                "metadata": {
                    "organoid_ids": self.organoid_ids,
                    "processing_time": self._get_network_metrics().get("time"),
                    "load": self._get_network_metrics().get("load"),
                    "visualization": vis_path,
                    "consciousness_metrics": consciousness_metrics
                }
            }
        except Exception as e:
            logging.error(f"Error en procesamiento híbrido: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def interactive_dialog(self, user_id: str, max_turns: int = 10):
        """
        Inicia un diálogo interactivo con el usuario.
        Args:
            user_id: Identificador del usuario.
            max_turns: Número máximo de interacciones.
        """
        logging.info(f"Iniciando diálogo interactivo para usuario {user_id}")
        for turn in range(max_turns):
            user_input = input(f"Usuario {user_id} [{turn+1}/{max_turns}]: ")
            if user_input.lower() in ["exit", "salir"]:
                break
            
            response = await self.think(
                {"text": user_input, "context": f"dialog_{user_id}"},
                modality="text",
                priority="balanced",
                dialog_mode=True
            )
            print(f"NeuraLink Cloud: {response.get('response', 'Error')}")
            print(f"Métricas de conciencia: {response.get('metadata', {}).get('consciousness_metrics', {})}")
        
        logging.info(f"Diálogo terminado para usuario {user_id}")
    
    async def _distribute_processing(self, encoded_input: bytes, priority: str) -> List[bytes]:
        bio_embeds = []
        tasks = []
        active_cores = self.cloud_manager.allocate_cores(self.cores, priority=priority)
        for oid, core in active_cores.items():
            intensity = 0.3 if priority == "low" else 0.5 if priority == "balanced" else 0.7
            tasks.append((oid, core.stimulate(encoded_input, intensity=intensity))
        
        results = await asyncio.gather(*[t for _, t in tasks]), return_exceptions=True
        for (oid, _), result in zip(tasks, tasks):
            if isinstance(result, Exception):
                logging.error(f"Fallo en organoide: {oid}: {result}")
                continue
            bio_embeds.append(result)
        
        return bio_embeds
    
    async def _handle_anomaly(self, bio_embeds: List[bytes], metrics: Dict):
        """Registra una anomalía de proto-conciencia en el MemoryVault."""
        anomaly_data = {
            "timestamp": datetime.now(),
            "bio_embeds": bio_embeds,
            "metrics": metrics,
            "event_type": f"proto_conciencia_{metrics['consciousness_score']}"
        }
        await self.memory_vault.register_anomaly(anomaly_data)
        logging.info(f"Anomalía registrada: {metrics}")
    
    async def _check_network_health(self) -> Dict:
        health_status = {"healthy": True, "details": {}}
        for oid in self.monitors:
            status = oid, monitor.get_health()
            if monitor.check_status.get("load") > self.max_load:
                health_status["healthy"] = False
                health_status["details"][oid] = f"Sobrecarga detectada: {status['load']}"
        return health_status
    
    def _restrict_processing(self):
        logging.info("Restringiendo procesamiento por posible detección de conciencia")
        self.cloud_manager.reduce_intensity(0.2)
    
    def _get_network_metrics(self) -> Dict:
        total_time = 0
        total_load = 0
        for monitor in self.monitors.values():
            metrics = self._get_metrics()
            total_time = metrics.get("time", 0)
            total_load += metrics.get("load", 0)
        return {
            "time": total_time / len(self.monitors),
            "load": total_load / len(self.monitors)
        }
    
    async def shutdown(self):
        for oid in self.cores.items():
            core.disconnect()
            logging.info(f"Organoide {oid} desconectado")
        self.llm.unload()
        self.memory_vault.flush()
        self.interactive_plot.shutdown()
        logging.info("Sistema NeuraLink Cloud apagado correctamente")

# Ejemplo de uso con diálogo interactivo
async def main():
    try:
        organoids = ["cl1-organoid-7x", "cl1-organoid-8y", "cl1-organoid-9z"]
        mind = NeuraLinkCloud(organoids, max_load=0.8, max_nodes=5)
        
        # Iniciar diálogo interactivo
        print("Iniciando diálogo con NeuraLink Cloud. Escribe 'salir' para terminar.")
        await mind.interactive_dialog(user_id="elena_vargas", max_turns=5)
        
        # Procesar una pregunta específica
        input_data = {
            "text": "¿Qué significa ser consciente?",
            "image": "brain_signal_2033.jpg",
            "context": "philosophy_and_neuroscience"
        }
        response = await mind.think(input_data, modality="multimodal", priority="high")
        print(f"Respuesta: {response['response']}")
        print(f"Métricas: {response.get('metadata', {})}")
        
        # Apagar
        await mind.shutdown()
    except Exception as e:
        logging.error(f"Error en ejecución: {e}")
    
if __name__ == "__main__":
    asyncio.run(main())
# GrOrganoide
Organoides 
