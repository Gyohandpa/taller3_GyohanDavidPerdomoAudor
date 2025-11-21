from dicom_app import DICOMApp

def main():
    app = DICOMApp(datos_dir="datos", dicom_subdir="datos_dicom", csv_name="resultado_dicom.csv")

    # 1) buscar dicoms
    try:
        app.scan_dicoms()
    except Exception as e:
        print(f"Error durante el escaneo de DICOM: {e}")
        return

    # 2) listar (muestra los primeros 100 para no saturar)
    print("\nArchivos DICOM encontrados (muestra hasta 100):")
    app.list_dicoms(show_n=100)

    # 3) elegir por número (opción A)
    total = len(app.dicom_files)
    while True:
        sel = input(f"\nSeleccione la imagen por número (1-{total}, o 'q' para salir): ").strip()
        if sel.lower() == "q":
            print("Proceso cancelado por el usuario.")
            return
        if not sel.isdigit():
            print("Entrada inválida. Ingrese un número.")
            continue
        idx = int(sel)
        if idx < 1 or idx > total:
            print("Número fuera de rango.")
            continue
        try:
            app.mostrar_detalle_y_mostrar_cortes(idx)
        except Exception as e:
            print(f"Error mostrando la imagen: {e}")
        break

if __name__ == "__main__":
    main()