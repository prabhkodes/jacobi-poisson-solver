reset
set term gif animate delay 10 optimize size 800,600
set output "jacobi.gif"

set title "Jacobi Iterations (stitched r0 + r1)"
set xlabel "Columns"
set ylabel "Global Rows"
unset key
set view map
set pm3d map
set palette rgbformulae 22,13,-31
set colorbox

data_dir = "files"

# Steps = count of r0 files
nsteps = system("ls ".data_dir."/jac_*_r0.dat 2>/dev/null | wc -l")

do for [s=1:nsteps] {
    tmp = sprintf("%s/_tmp_%04d_all.dat", data_dir, s)

    # Concatenate r0 then r1 for this step → one full matrix
    system(sprintf("cat %s/jac_%04d_r0.dat %s/jac_%04d_r1.dat > %s", \
                   data_dir, s, data_dir, s, tmp))

    # Plot as a matrix heatmap
    splot tmp matrix
}

unset output
