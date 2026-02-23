set terminal png size 1500, 1000 font "Sans,22"

set xlabel "MJD"
set xrange [:]

set ylabel "Радиальная скорость, км/с"
set yrange [:]

set grid
set key top right box opaque

set fit nolog

a = 100.0
b = 0.02120791819
f(x) = a * sin(b * x + c)
fit f(x) "data/rad_vel_err.dat" via a, c

set output "rad_vel_err.png"

plot "data/rad_vel_err.dat" with points ls 5 lc rgb 'black' title "Радиальные скорости объекта", \
    f(x) with line ls 5 lc rgb 'red' title "y = a * sin(b * x + c)"
