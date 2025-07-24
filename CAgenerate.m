function [CAcodehigh, CAcodelow] = CAgenerate(PRNnum)
%UNTITLED 生成LEO的CA码
for k=1:1:2 
    [g2_initialphase,fisrt24,end24]=CABIAO(PRNnum,k-1);
    g1=ones(1,11);

    for i=1:2046
        g1output(i)=g1(11);
        temp=mod(g1(1)+g1(7)+g1(8)+g1(9)+g1(10)+g1(11),2);
        g1(2:11)=g1(1:10);
        g1(1)=temp;  
    end
    g2=g2_initialphase;
    for i=1:2046
        g2output(i)=g2(11);
        temp=mod(g2(1)+g2(3)+g2(8)+g2(9)+g2(10)+g2(11),2);
        g2(2:11)=g2(1:10);
        g2(1)=temp;  

    end
    for i=1:2046
        CAcode(i)=2*(mod(g1output(i)+g2output(i),2))-1;    
    end

    %check是否正确  resultcheck=0则正确
    resultcheck=24;
    for i=1:1:24
        fisrt24check(i)=CAcode(i)-fisrt24(i);
        end24check(i)=CAcode(2046-24+i)-end24(i);
        if(fisrt24check(i)==0)
            if(end24check(i)==0)
                resultcheck=resultcheck-1;
            end
        end
    end
    if(resultcheck~=0)
       disp("error for CAgenerate") ;
    else
       disp("successful for CAgenerate") ;
       if((k-1)==0)
           CAcodelow=CAcode;
       else
           CAcodehigh=CAcode;
       end
    end
end
end