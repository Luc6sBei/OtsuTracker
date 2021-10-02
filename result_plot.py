import matplotlib.pyplot as plt
import numpy as np
import datetime
import csv

RELATIVE_DESTINATION_PATH = str(datetime.date.today()) + '_result/'
total_error_array = np.zeros(0)
error_array = np.zeros(0)
speed_array = np.zeros(0)
time_array = np.zeros(0)
dis_array = np.zeros(0)

with open(RELATIVE_DESTINATION_PATH + "results.csv", "r") as csvFile:
    reader = csv.reader(csvFile)
    rows = [row for row in reader]

for n in range(1, len(rows)):
    time_array = np.append(time_array, float(rows[n][1]))
    dis_array = np.append(dis_array, float(rows[n][2]))
    error_array = np.append(error_array, float(rows[n][6]))
    speed_array = np.append(speed_array, float(rows[n][3]))
    total_error_array = np.append(total_error_array, float(rows[n][7]))


# plot the total error
save_path = 'image/Otsu_'
plt.plot(time_array[1::], total_error_array[1::])
plt.xlabel('Time(s)')
plt.ylabel('Total error')
plt.title('Total error plot of the Otsu method')
plt.savefig(save_path+'totalError')
plt.show()
# plot the error
plt.plot(time_array[1::], error_array[1::])
plt.xlabel('Time(s)')
plt.ylabel('Error')
plt.title('Error plot of the Otsu method')
plt.savefig(save_path+'error')
plt.show()
plt.plot(time_array[1::], speed_array[1::])
plt.xlabel('Time(s)')
plt.ylabel('Speed(unit/s)')
plt.title('Speed plot of the Otsu method')
plt.savefig(save_path+'speed')
plt.show()
plt.plot(time_array[1::], dis_array[1::])
plt.xlabel('Time(s)')
plt.ylabel('Distance(unit)')
plt.title('Distance plot of the Otsu method')
plt.savefig(save_path+'distance')
plt.show()
