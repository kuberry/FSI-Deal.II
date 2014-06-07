mkdir $1
cp FSI_Project $1
cp default.prm $1
cp ts.py $1
cd $1
python ts.py
