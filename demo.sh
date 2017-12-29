#!/bin/bash
values=(-0.4 -0.3 -0.2 -0.1 0 0.1 0.2 0.3 0.4)
allcodes=("600176" "002594" "000725" "600582" "600050" "600036" "002456" "002415")

valuelength=${#values[@]}
codelength=${#allcodes[@]}
# echo ${values[1]} ${#allcodes[1]}
# for ((i=0; i<=$valuelength; i ++))  
# do  
    for ((j=1; j<=10; j++))  
    do  
        for ((k=0; k<=$codelength; k++))  
        do  
            val=`expr $j \* 10`
            echo  ${values[1]}  ${allcodes[$k]}  $val
            python gossip.py  ${values[1]}  ${allcodes[$k]}  $val
        done    
    done    
# done  
