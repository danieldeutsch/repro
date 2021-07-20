# https://stackoverflow.com/questions/929368/how-to-test-an-internet-connection-with-bash
wget -q --spider http://google.com

if [ $? -eq 0 ]; then
    echo "Online" > /tmp0/results.txt
else
    echo "Offline" > /tmp0/results.txt
fi