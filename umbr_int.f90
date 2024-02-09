!==============================================================
!
!  UMBR_INT v. 1.2
!
!==============================================================
!
!  Performs "umbrella integration" for umbrella sampling
!  sinulations. 
!  
!  The method:
!
!  Johannes Kästner and Walter Thiel, "Bridging the gap between 
!  thermodynamic integration and umbrella sampling provides a novel 
!  analysis method: "Umbrella integration", 
!  J. Chem. Phys. 123, 144104 (2005)
!
!  This program only reads GROMACS 4.* files!
!  The code should be modified to read any other format.
!
!==============================================================
!  (C) Semen Yesylevskyy, 2007
!==============================================================


program umb_int
implicit none

 REAL(8),PARAMETER :: two_pi=6.283185307179586476925286766559005768394
 
 real(8),parameter ::  kb = 0.008296553 ! Previous value was incorrect!!!!!!
 
 !real(8),parameter ::  kb = 0.001982923700 ! Same as in wham
 
 ! Performs umbrella integration for umbrella sampling
 integer :: Nbin, Nwin
 real(8) :: bin_min,bin_max,bin_sz
 integer :: i,j,ios,i1,i2
 integer,dimension(:),allocatable :: num
 character(100) :: fname,dum
 integer :: d
 real(8),dimension(:),allocatable :: x0,k ! Centers and constants
 real(8),dimension(:),allocatable :: mean,std  ! Params of individual windows
 real(8),dimension(:,:),allocatable :: dAu ! For individual windows
 real(8),dimension(:),allocatable :: dAfinal,Afinal
 real(8) :: Temperature
 
 real(8) :: time,temp,sqrt_two_pi,x,x_ref
 real(8) :: ref_coor,coor
 
 sqrt_two_pi = sqrt(two_pi)

 !TO BE DEFINED!

 ! Read number of bins on reaction coordinate
 read(*,*) Nbin
 print *,"Will use ",Nbin," bins"
 ! Read dimensions
 read(*,*) bin_min
 read(*,*) bin_max
 print *,"Reaction cordinate from ",bin_min," to ",bin_max
 bin_sz = (bin_max-bin_min)/real(Nbin)
 
 ! Read number of windows
 read(*,*) Nwin
 print *,"Will read ",Nwin," windows"
 
 ! Read temperature
 read(*,*) Temperature
 print *,"At temperature ",Temperature
 
 allocate( x0(1:Nwin), k(1:Nwin), mean(1:Nwin), std(1:Nwin) )
 allocate( dAu(1:Nwin,1:Nbin) )
 allocate( dAfinal(1:Nbin), Afinal(1:Nbin) )
 allocate( num(1:Nbin) )
 
 ! Read window files
 
 do i=1,Nwin
   ! Gromacs 4.x does not write x0 and k to output file, so it should be provided by user
   ! Read x0 and k
   read(*,*) x0(i)
   read(*,*) k(i)
      
   ! Read name of file
   read(*,"(A)") fname  
   print *,"Window ",i," Center: ",x0(i)," Constant: ",k(i)," Opening file ",trim(fname)
   open(111,file=trim(fname),action="read",iostat=ios)
   if(ios/=0) then
     print *,"Error reading file ",fname
     stop
   end if 
   
   ! Skip header
   do 
    read(111,*) dum
    if(dum(1:1)/="#" .and. dum(1:1)/="@")then
     exit
    end if
   end do
   ! Go one line up to read first line again
   backspace(111)
   
   ! Reading data

   print *,"Reading data..."
   num(i) = 0
   mean(i) = 0.0
   temp = 0.0
   do
     read(111,*,iostat=ios) time,x
     if(ios==-1)then 
      exit
     end if
     
     !x = coor-ref_coor
     ! Put to bin
     !x = x+x0(i)
     !x = coor-ref_coor !+x0(i)
     
     i1 = get_bin(x)
     
     if(i1<=Nbin .and. i1>=1)then
      num(i) = num(i)+1
      mean(i) = mean(i) + x
      temp = temp + x*x
     end if
   end do
   print *,"Read in ",num(i)," values"
   
   if(num(i)==0)then
    print *,"ERROR! All values from this window are outside the specified bonds."
    stop
   end if
   
   mean(i) = mean(i)/real(num(i))
   temp    = temp/real(num(i))
   std(i) = sqrt(temp-mean(i)*mean(i))

   print *,"Mean: ",mean(i)," std: ",std(i)
   
   close(111)
   
 end do
 
 print *,"Reading finished."
 print *,"Computing dAu..."
 
 do i=1,Nwin
   do j=1,Nbin
     x = bin_min + bin_sz*(j-0.5)  ! bin back to absolute x
     dAu(i,j) = (kb*Temperature * (x-mean(i))/(std(i)*std(i)) ) - k(i)*(x-x0(i))
   end do
 end do
 
 print *,"Combining windows..."

 do j=1,Nbin
   x = bin_min + bin_sz*(j-0.5)  ! bin back to absolute x
   dAfinal(j) = 0.0

   ! Get weight normalization
   temp = 0.0
   do i=1,Nwin
     temp = temp + num(i)*Pb(i,x)
   end do

   do i=1,Nwin
     dAfinal(j) = dAfinal(j) + (num(i)*Pb(i,x)/temp)*dAu(i,j)
   end do
 end do
 
 print *,"Integrating..."
 Afinal(1) = 0.0
 do j=2,Nbin
   Afinal(j) = Afinal(j-1) + bin_sz*0.5*(dAfinal(j-1)+dAfinal(j))
 end do
 
 ! Shift to make 0 at bottom
 temp = minval(Afinal)
 Afinal = Afinal - temp
 
 print *,"Writing output..."
 
 open(111,file="free_energy.dat")
 do j=1,Nbin
  write(111,*) bin_min+bin_sz*j-0.5*bin_sz,Afinal(j)
 end do
 
contains

 function get_bin(x)
  real(8) :: x
  integer :: get_bin
  
  get_bin = int((x - bin_min)/bin_sz) + 1
 end function
 
 function Pb(w,x)
  integer :: w
  real(8) :: x,Pb,s
  Pb = 1/(std(w)*sqrt_two_pi) * exp( -0.5*(( (x-mean(w))/std(w) )**2.0) )
 end function

end program

