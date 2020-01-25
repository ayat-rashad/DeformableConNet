# setup (python2) environment
pip2 install -r requirements.txt

# setup deformable convolution operator
cd deform_conv
bash prepare.sh
cd ..

# test 1,2,3 deformable layers
python main.py --layers=1
python main.py --layers=2
python main.py --layers=3
