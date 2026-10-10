# Contribuir

1. Abrir una rama descriptiva.
2. No añadir videos de la prueba, pesos, bases de Label Studio ni secretos.
3. Añadir o actualizar tests para cada cambio de contrato.
4. Ejecutar `python scripts/validate_publication.py` y la suite completa.
5. Describir supuestos, limitaciones y origen de cualquier métrica.

Los cambios que afecten seguridad, geometría o clasificación deben incluir un
caso negativo que demuestre que el sistema falla de forma segura.
