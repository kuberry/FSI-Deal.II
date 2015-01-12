set terminal postscript eps color lw 4 "Times-Roman" 18
#set terminal fig color # lw 3 "Times-Roman" 14
#set term epslatex color lw 3 "Times-Roman" 14
set key off
set size ratio 1.0
set xtics .05 font "Times-Roman,16"
set ytics .05
set tmargin 1
set bmargin 1
set lmargin 7
set rmargin 1
set key on 
set output './DNvsBICG.eps'
set multiplot layout 1,3
set border lw 0.5
set title "(1.5,1.0)" font "Times-Roman,22"
set ylabel "Displacement (cm)" offset 2
set key bottom center samplen 2 invert
set style line 1 lc rgb "#FF0000" lt 1 lw .5
set style line 2 lc rgb "#00FF00" lt 1 lw .5
plot "./DN.txt" every ::1 u 1:2 title "DN" w l ls 1,\
 "./BICG.txt" every ::1 u 1:2 title "BICG" w l ls 2
set title "(3.0,1.0)" font "Times-Roman,22"
unset ylabel
#set ylabel "Displacement (cm)"
set xlabel "Time (seconds)"
set key bottom right samplen 2
plot "./DN.txt" every ::1 u 1:3 title "DN" w l ls 1,\
 "./BICG.txt" every ::1 u 1:3 title "BICG" w l ls 2
set title "(4.5,1.0)" font "Times-Roman,22"
#set ylabel "Displacement (cm)"
unset xlabel
set key top left samplen 2
plot  "./DN.txt" every ::1 u 1:4 title "DN" w l ls 1,\
 "./BICG.txt" every ::1 u 1:4 title "BICG" w l ls 2
unset multiplot
set lmargin at screen 0.05
set bmargin at screen 0.05
set rmargin at screen 0.95
set tmargin at screen 0.95
