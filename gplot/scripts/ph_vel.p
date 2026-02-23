set terminal png size 1500, 1000 font "Sans,22"

set xlabel "Фаза"
set xrange [:]

set ylabel "Радиальная скорость, км/с"
set yrange [:]

set grid
set key top right box opaque

set fit nolog

f(x) = g + k * sin(2 * pi * (x - p))
fit f(x) "data/ph_vel.dat" via g, k, p

set output "ph_vel.png"

plot "data/ph_vel.dat" with points ls 5 lc rgb 'black' title "Радиальные скорости объекта", \
    f(x) with line ls 5 lc rgb 'red' title "y = g + k * sin(2 * pi * (x - p))"
