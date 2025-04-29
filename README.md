# cse472-assignment-2-hidden-markov-models-hmm-solved
**TO GET THIS SOLUTION VISIT:** [CSE472 Assignment 2-Hidden Markov Models (HMM) Solved](https://www.ankitcodinghub.com/product/cse472-machine-learning-sessional-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;112866&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CSE472 Assignment 2-Hidden Markov Models (HMM) Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
Introduction

In this assignment, you will learn modeling with a Hidden Markov Model (HMM), and implement the Viterbi and the Baum-Welch algorithm. HMMs are applied in many fields including speech and image processing, bioinformatics and finance.

Climate pattern modeling

El Nin˜o–Southern Oscillation (ENSO) is an irregular periodic variation in winds and sea surface temperatures over the tropical eastern Pacific Ocean, affecting the climate of much of the tropics and subtropics. The warming phase of the sea temperature is known as El Nin˜o and the cooling phase as La Nin˜a both of which have long-term persistence. El Nin˜o years in a particular basin tend to be wetter than La Nin˜a years.

We can observe whether or not it is an El Nin˜o year based on rainfall in the tropical Pacific for the present. But we are interested in understanding past climate variation using tree ring widths. We can infer from the tree ring data (with some error) what the total precipitation might have been in each year of the tree’s life.

So, we have two hidden states representing El Nin˜o and La Nin˜a. The observed quantities are rainfall estimates (from tree ring width) for the past T years. Let’s assume for simplicity that our observations Yt can be modeled by Gaussian distributions i.e. ) and f(Yt|Xt = 2) ∼

Dataset

You will be given two files titled “data.txt” and “parameters.txt” containing the rainfall data for the past T years and parameters for the HMM respectively.

Input

• The “data.txt” file contains T rows each containing a number indicating the rainfall for that year.

• The “parameters.txt” file contains the number of states n (2 in this case) in the first line. The next n lines provides the transition matrix P. The next line gives the means of the n Gaussian distributions and the last line lists the standard deviations of the n Gaussian distributions.

Note: Your implementations must be easy to extend to arbitrary number of states of and emission probability distributions.

Output

• A file containing estimated states using the parameters provided in “parameters.txt”. This will contain T rows each containing the estimated state for that year. • A file containing parameters learned using the Baum-Welch algorithm. The format will be the same as “parameters.txt”. Add the stationary distribution in the last line.

• A file containing estimated states using the learned parameters. This will contain T rows each containing the estimated state for that year.

Viterbi algorithm implementation

In this case, you will be given

1. The parameters of the HMM i.e. the transition matrix P, the initial probabilities of the states π, and the parameters for the Gaussian distributions µ1,σ1,µ2,σ2 in the “parameters.txt” file. (Use the stationary distribution of P as the initial probabilities. https://www. stat.berkeley.edu/~mgoldman/Section0220.pdf)

2. The rainfall estimates y1,y2,…yT in “data.txt”

Your task is to

• Implement the Viterbi algorithm to estimate the most likely hidden state sequence x1,x2,…xT (El Nin˜o or La Nin˜a) for the past T years.

Baum-Welch implementation

Now we will also estimate the parameters of the HMM. Given,

• The rainfall estimates y1,y2,…yT in “data.txt”

You will implement the Baum-Welch algorithm to estimate

1. The most likely values of parameters of the HMM i.e. the transition matrix P, and the parameters for the Gaussian distributions µ1,σ1,µ2,σ2. You can assume the initial probabilities π to be the stationary distribution of P.

2. The most likely hidden state sequence x1,x2,…xT (El Nin˜o or La Nin˜a) for the past T years for the finally estimated parameters.

The Baum–Welch algorithm is a special case of the Expectation-Maximization (EM) algorithm used to find the unknown parameters of a hidden Markov model (HMM). Expectation-Maximization is a two-step process for maximum likelihood estimation when the likelihood function cannot be computed directly, for example, because its observations are hidden as in an HMM.

For this, initialize the parameters P,µ1,σ1,µ2,σ2 with random values (save the seed). Then iterate the following two until convergence:

1. E-step: Use the forward-backward equations to find the expected hidden states given the observed data and the set of current parameter values

2. M-step: This is the update phase. In this step, find the parameter values that best fit the expected hidden states given the observed data.

Note: You can use the values provided in “parameters.txt” for initialization if there are convergence problems.

Outputs

You will output the estimated hidden state sequence and parameter values to files. The details will be provided later. Also, compare the solution results with the sci-kit hmmlearn.

Submission

2. Write code in a single *.py file, then rename it with your student id. For example, if your student id is 1605123, then your code file name should be “1605123.py” and the report name should be “1605123.pdf”.

3. Finally make a main folder, put the code and report in it, and rename the main folder as your student id. Then zip it and upload it.

Evaluation

1. You have to reproduce your experiments during in-lab evaluation. Keep everything ready to minimize delay.

2. You are likely to give online tasks during evaluation which will require you to modify your code.

3. You will be tested on your understanding through viva-voce.

5. You are encouraged to bring your computer in the sessional to avoid any hassle. But in that case, ensure an internet connection as you have to instantly download your code from the Moodle and show it.

1. Don’t copy! We regularly use copy checkers.

2. First time copier and copyee will receive negative marking because of dishonesty. Their default is bigger than those who will not submit.

3. Repeated occurrence will lead severe departmental action and jeopardize your academic career. We expect fairness and honesty from you. Don’t disappoint us!
