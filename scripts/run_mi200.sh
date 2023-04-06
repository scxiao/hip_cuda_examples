
blocknum=4096
loc=.
for i in {2..31}
do
    blocksize=$(($i*512))
    cmd="$loc/test_layernorm_fuse $blocknum $blocksize"
    echo $cmd
    $cmd
done
