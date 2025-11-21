import os
import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DICOMApp:
    def __init__(self, datos_dir="datos", dicom_subdir="datos_dicom", csv_name="resultado_dicom.csv"):
        # Rutas
        self.base_path = os.path.abspath(datos_dir)
        self.csv_path = os.path.join(self.base_path, csv_name)
        self.dicom_path = os.path.join(self.base_path, dicom_subdir)

        # Datos
        self.csv = None
        self.dicom_files = []
        self.series = []
        self.volume = None

        self._validate_structure()
        self._load_csv()

    # ==========================================
    # VALIDACIONES Y LECTURA DE CSV
    # ==========================================

    def _validate_structure(self):
        if not os.path.exists(self.base_path):
            raise FileNotFoundError(f"No existe la carpeta {self.base_path}")
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError("No se encontró resultado_dicom.csv dentro de la carpeta datos")
        if not os.path.exists(self.dicom_path):
            raise FileNotFoundError("No se encontró carpeta datos_dicom")

    def _load_csv(self):
        self.csv = pd.read_csv(self.csv_path)

    # ==========================================
    # ESCANEO Y LISTA DE DICOM
    # ==========================================

    def scan_dicoms(self):
        self.dicom_files = [
            os.path.join(self.dicom_path, f)
            for f in os.listdir(self.dicom_path)
            if f.lower().endswith(".dcm")
        ]
        if not self.dicom_files:
            raise FileNotFoundError("No se encontraron .dcm en datos_dicom")

        registros = []  # cada fila del DataFrame

        for f in self.dicom_files:
            try:
                ds = pydicom.dcmread(f)

                # Cálculo de intensidad promedio
                pixel_mean = float(np.mean(ds.pixel_array)) if hasattr(ds, "pixel_array") else np.nan

                # Registro de metadatos
                registros.append({
                    "PacienteID": getattr(ds, "PatientID", "Desconocido"),
                    "PacienteNombre": getattr(ds, "PatientName", "Desconocido"),
                    "EstudioUID": getattr(ds, "StudyInstanceUID", "Desconocido"),
                    "Descripcion": getattr(ds, "StudyDescription", "Desconocido"),
                    "FechaEstudio": getattr(ds, "StudyDate", "Desconocido"),
                    "Modalidad": getattr(ds, "Modality", "Desconocido"),
                    "Filas": getattr(ds, "Rows", np.nan),
                    "Columnas": getattr(ds, "Columns", np.nan),
                    "IntensidadPromedio": pixel_mean
                })

            except Exception:
                pass

        # === Construir DataFrame Estructurado (REQUISITO 4.3) ===
        self.csv = pd.DataFrame(registros)

        # === Guardar DataFrame en resultado_dicom.csv (REQUISITO 4.4) ===
        self.csv.to_csv(self.csv_path, index=False)

        print("\nArchivo resultado_dicom.csv actualizado correctamente.")

    def list_dicoms(self, show_n=100):
        for i, f in enumerate(self.dicom_files[:show_n], start=1):
            print(f"{i}: {os.path.basename(f)}")

    # ==========================================
    # RECONSTRUCCIÓN DE VOLUMEN
    # ==========================================

    def _load_series(self):
        ds_list = []
        for f in self.dicom_files:
            try:
                ds = pydicom.dcmread(f)
                if hasattr(ds, "SeriesInstanceUID"):
                    ds_list.append(ds)
            except:
                pass

        # Agrupar por UID
        series_dict = {}
        for ds in ds_list:
            uid = getattr(ds, "SeriesInstanceUID", None)
            series_dict.setdefault(uid, []).append(ds)

        # Ordenar: la serie más grande será usada
        self.series = sorted(series_dict.values(), key=lambda s: len(s), reverse=True)
        if not self.series:
            raise ValueError("No se encontraron series válidas.")

    def _build_volume(self, serie):
        # Ordenar cortes
        try:
            serie = sorted(serie, key=lambda s: float(s.ImagePositionPatient[2]))
        except:
            serie = sorted(serie, key=lambda s: int(s.InstanceNumber))

        slices = []
        for s in serie:
            try:
                slices.append(s.pixel_array.astype(np.float32))
            except:
                pass

        vol = np.stack(slices, axis=0)

        # =======================
        # Corrección de proporción
        # =======================

        thk = getattr(serie[0], "SliceThickness", 1.0)
        px_spacing = getattr(serie[0], "PixelSpacing", [1.0, 1.0])

        try:
            thk = float(thk)
        except:
            thk = 1.0

        try:
            px_y, px_x = map(float, px_spacing)
        except:
            px_y, px_x = 1.0, 1.0

        scale_z = thk / px_y

        if scale_z > 0:
            target_z = int(vol.shape[0] * scale_z)
            vol = np.stack([
                vol[int(i / scale_z)] if int(i / scale_z) < vol.shape[0] else vol[-1]
                for i in range(target_z)
            ], axis=0)

        # Normalizar
        vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol)) * 255
        return vol.astype(np.uint8), serie[0]

    # ==========================================
    # MOSTRAR RESULTADOS
    # ==========================================

    def mostrar_detalle_y_mostrar_cortes(self, index):
        if index < 1 or index > len(self.dicom_files):
            raise ValueError("Índice fuera de rango")

        # Generar volumen si no existe
        if self.volume is None:
            self._load_series()
            self.volume, ref = self._build_volume(self.series[0])
        else:
            # Usar mismo volumen pero referencia adecuada
            ref = pydicom.dcmread(self.dicom_files[index - 1])

        # Mostrar metadatos requeridos
        self._show_metadata(ref, index)

        # Mostrar cortes
        self._show_planes()

    def _show_metadata(self, ds, index):
        print("\n===== METADATOS DEL ESTUDIO =====")
        print("Identificador del paciente:", getattr(ds, "PatientID", "Desconocido"))
        print("Nombre del paciente:", getattr(ds, "PatientName", "Desconocido"))
        print("Identificador único del estudio:", getattr(ds, "StudyInstanceUID", "Desconocido"))
        print("Descripción del estudio:", getattr(ds, "StudyDescription", "Desconocido"))
        print("Fecha del estudio:", getattr(ds, "StudyDate", "Desconocido"))
        print("Modalidad:", getattr(ds, "Modality", "Desconocido"))
        print("Número de filas:", getattr(ds, "Rows", "Desconocido"))
        print("Número de columnas:", getattr(ds, "Columns", "Desconocido"))
        
        # NUEVA LÍNEA (4.4 adicional)
        if self.csv is not None and "IntensidadPromedio" in self.csv.columns:
            try:
                valor = self.csv.iloc[index - 1]["IntensidadPromedio"]
                print("Intensidad promedio:", valor)
            except:
                print("Intensidad promedio: No disponible")

    def _show_planes(self):
        vol = self.volume
        z, h, w = vol.shape
        axial = vol[z // 2, :, :]
        coronal = vol[:, h // 2, :]
        sagittal = vol[:, :, w // 2]

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(axial, cmap="gray"); axs[0].set_title("Axial"); axs[0].axis("off")
        axs[1].imshow(coronal, cmap="gray"); axs[1].set_title("Coronal"); axs[1].axis("off")
        axs[2].imshow(sagittal, cmap="gray"); axs[2].set_title("Sagital"); axs[2].axis("off")
        plt.tight_layout()
        plt.show()