# from concurrent.futures import ThreadPoolExecutor
# import os
# import time

# def dormir(tiempo):
#     print(f"Hilo {os.getpid()} durmiendo por {tiempo} segundo(s).")
#     time.sleep(tiempo)
#     print(f"Hilo {os.getpid()} despertó.")

# with ThreadPoolExecutor(max_workers=2) as executor:
#     executor.submit(dormir, 1)
#     executor.submit(dormir, 1)
#     executor.submit(dormir, 1)


# from concurrent.futures import ProcessPoolExecutor
# import os
# import time

# def dormir(tiempo):
#     print(f"Proceso {os.getpid()} durmiendo por {tiempo} segundo(s).")
#     time.sleep(tiempo)
#     print(f"Proceso {os.getpid()} despertó.")

# def main():
#     with ProcessPoolExecutor(max_workers=2) as executor:
#         executor.submit(dormir, 1)
#         executor.submit(dormir, 1)

# if __name__ == '__main__':
#     main()

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time

def productor(col):
    # Simula alguna operación
    time.sleep(1)
    col.put("Datos desde el productor")

def consumidor(col):
    # Espera a recibir datos
    datos = col.get()
    print(f"El consumidor recibió: {datos}")

if __name__ == '__main__':
    col = multiprocessing.Queue()
    with ProcessPoolExecutor(max_workers=2) as executor:
        executor.submit(productor, col)
        executor.submit(consumidor, col)


