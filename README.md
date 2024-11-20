# PIA_IA
INTRODUCCION

El reconocimiento y la clasificación de placas de vehículos son tareas fundamentales en diversas aplicaciones de seguridad y gestión del tráfico, como el control de acceso en estacionamientos, sistemas de peaje automático y vigilancia en carreteras. El objetivo de este proyecto es desarrollar un sistema capaz de detectar y leer placas vehiculares en tiempo real utilizando técnicas avanzadas de visión por computadora y aprendizaje automático. Para ello, se emplearán dos tecnologías clave: OCR (Reconocimiento Óptico de Caracteres) con PaddlePaddle y detección de objetos con YOLO (You Only Look Once).

El uso combinado de PaddleOCR y YOLO permite aprovechar lo mejor de ambos mundos: mientras que YOLO se encarga de la detección precisa y rápida de las placas dentro de las imágenes, PaddleOCR se especializa en el reconocimiento y la extracción de los caracteres alfanuméricos presentes en esas placas. Este enfoque no solo mejora la exactitud del sistema, sino que también optimiza su capacidad para trabajar en tiempo real, un requisito crucial para aplicaciones en vivo.

El sistema propuesto estará basado en un flujo de trabajo eficiente que integrará la captura de imágenes desde una cámara en tiempo real, el procesamiento y la detección de placas mediante YOLO, y la posterior lectura de los caracteres utilizando PaddleOCR. La implementación de este sistema busca ofrecer una solución robusta, precisa y rápida, capaz de operar en condiciones variables como diferentes tipos de iluminación, distancias o ángulos de captura.

Este proyecto no solo busca mejorar los métodos actuales de reconocimiento de placas vehiculares, sino también contribuir al desarrollo de aplicaciones inteligentes que puedan integrar fácilmente estos avances tecnológicos para optimizar la gestión del tráfico y la seguridad vial.
A lo largo de este proyecto estaremos utilizando esta librería junto con otras para el reconocimiento de caracteres usando visión por computadora.

